import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import OurNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, DomainContrastiveLoss, target2onehot
from backbone.vit_ours import Adapter_lora
import math
import random
import os 
num_workers = 8


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def compute_orthogonality_loss(previous_weights_list, current_weights, epsilon=1e-8):
    total_ortho_loss = 0.0
    current_norm = torch.norm(current_weights.flatten())
    current_normalized = current_weights.flatten() / (current_norm + epsilon)

    for prev_weights in previous_weights_list:
        # Normalize previous weights
        prev_norm = torch.norm(prev_weights.flatten())
        prev_normalized = prev_weights.flatten() / (prev_norm + epsilon)

        # Compute absolute dot product (should be close to 0 for orthogonal vectors)
        dot_product = torch.abs(torch.sum(prev_normalized * current_normalized))

        total_ortho_loss += dot_product

    # Average over all previous tasks
    if len(previous_weights_list) > 0:
        total_ortho_loss /= len(previous_weights_list)

    return total_ortho_loss

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = OurNet(args, True)

        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]

        self.use_exemplars = args["use_old_data"]
        self.use_init_ptm = args["use_init_ptm"]
        self.use_diagonal = args["use_diagonal"]

        self.recalc_sim = args["recalc_sim"]
        self.alpha = args["alpha"] # forward_reweight is divide by _cur_task
        self.beta = args["beta"]
        self.class_proto = torch.tensor([]).to(self._device)

        self.moni_adam = args["moni_adam"]
        self.adapter_num = args["adapter_num"]
        
        self.anti_prototype_loss = DomainContrastiveLoss(
            margin=args["margin"], 
            alpha=args["alpha_loss"], 
            beta=args["beta_loss"], 
            gamma=args["gamma"], 
            temperature=args["temperature"],
            adaptive_temp = args["adaptive_temp"],
            smooth_loss= args["smooth_loss"],
            curriculum_learning = args["curriculum_learning"]
            )
        if self.moni_adam:
            self.use_init_ptm = True
            self.alpha = 1
            self.beta = 1

    def after_task(self):
        self._known_classes = self._total_classes
        self._network.freeze()
        self._network.backbone.add_adapter_to_list()

    def get_cls_range(self, task_id):
        if self.task_increments:
            self.inc = self.task_increments[task_id]
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            start_cls = self.init_cls + (task_id - 1) * self.inc
            end_cls = start_cls + self.inc

        return start_cls, end_cls

    def replace_fc(self, train_loader):
        model = self._network
        model = model.eval()
        # 初始化原型置信度计算器
        min_samples = 3  # 最小有效样本数
        var_threshold = 0.15  # 方差阈值
        with torch.no_grad():
            # replace proto for each adapter in the current task
            # if self.args["dataset"] == "SKIN":
            #     # 获取历史原型（假设存储在self.hist_protos中）
            #     hist_protos = getattr(self, 'hist_protos', {})
            if self.use_init_ptm:
                start_idx = -1
            else:
                start_idx = 0

            for index in range(start_idx, self._cur_task + 1):
                if self.moni_adam:
                    if index > self.adapter_num - 1:
                        break
                # only use the diagonal feature, index = -1 denotes using init PTM, index = self._cur_task denotes the last adapter's feature
                elif self.use_diagonal and index != -1 and index != self._cur_task:
                    continue

                embedding_list, label_list = [], []
                for i, batch in enumerate(train_loader):
                    (_, data, label) = batch
                    data = data.to(self._device)
                    label = label.to(self._device)
                    embedding = model.backbone.forward_proto(data, adapt_index=index)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())

                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)

                class_list = np.unique(self.train_dataset_for_protonet.labels)
                # model.fc.weight.data[self._known_classes:self._total_classes, index*self._network.out_dim:(index+1)*self._network.out_dim] = self._network.fc_list[-1].weight.data
                # if self.args["dataset"] == "SKIN":
                #     current_protos = {}
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    
                    # # 计算当前类别的样本数量和特征方差
                    # n_samples = embedding.shape[0]
                    # variances = torch.var(embedding, dim=0)
                    # avg_variance = torch.mean(variances).item()
                    
                    # # 计算原型置信度
                    # sample_confidence = min(1.0, n_samples / (2 * min_samples)) if n_samples > 0 else 0.0
                    # var_confidence = max(0.0, 1.0 - avg_variance / var_threshold)
                    # confidence = math.sqrt(sample_confidence * var_confidence)
                    
                    proto = embedding.mean(0)
                    if index == self._cur_task:
                        self.class_proto = torch.cat((self.class_proto, proto.unsqueeze(0).to(self._device)), dim=0)
                    if self.use_init_ptm:
                        model.fc.weight.data[class_index, (index+1)*self._network.out_dim:(index+2)*self._network.out_dim] = proto
                    else:
                        model.fc.weight.data[class_index, index*self._network.out_dim:(index+1)*self._network.out_dim] = proto
        return
    
    # def replace_fc(self, train_loader):
    #     model = self._network
    #     model = model.eval()
        
    #     # 原型质量评估参数
    #     min_samples = 3       # 最小有效样本数
    #     var_threshold = 0.15  # 方差阈值
    #     global_center = None  # 全局特征中心
        
    #     with torch.no_grad():
    #         if self.use_init_ptm:
    #             start_idx = -1
    #         else:
    #             start_idx = 0

    #         for index in range(start_idx, self._cur_task + 1):
    #             if self.moni_adam:
    #                 if index > self.adapter_num - 1:
    #                     break
    #             elif self.use_diagonal and index != -1 and index != self._cur_task:
    #                 continue

    #             embedding_list, label_list = [], []
    #             for i, batch in enumerate(train_loader):
    #                 (_, data, label) = batch
    #                 data = data.to(self._device)
    #                 label = label.to(self._device)
    #                 embedding = model.backbone.forward_proto(data, adapt_index=index)
    #                 embedding_list.append(embedding.cpu())
    #                 label_list.append(label.cpu())

    #             embedding_list = torch.cat(embedding_list, dim=0)
    #             label_list = torch.cat(label_list, dim=0)
                
    #             # 计算全局特征中心（所有样本的均值）
    #             if global_center is None:
    #                 global_center = embedding_list.mean(dim=0)
    #             else:
    #                 # 指数移动平均更新全局中心
    #                 global_center = 0.7 * global_center + 0.3 * embedding_list.mean(dim=0)

    #             class_list = np.unique(self.train_dataset_for_protonet.labels)
                
    #             # 第一步：计算所有类别的原始原型
    #             raw_protos = {}  # 存储每个类别的原始原型
    #             class_counts = {}
    #             class_variances = {}
                
    #             for class_index in class_list:
    #                 data_index = (label_list == class_index).nonzero().squeeze(-1)
    #                 if len(data_index) > 0:
    #                     embedding = embedding_list[data_index]
    #                     # 计算原始原型
    #                     raw_proto = embedding.mean(0)
    #                     # 计算样本数量
    #                     count = embedding.shape[0]
    #                     # 计算特征方差
    #                     variances = torch.var(embedding, dim=0)
    #                     avg_variance = torch.mean(variances).item()
    #                 else:
    #                     raw_proto = global_center.clone()
    #                     count = 0
    #                     avg_variance = float('inf')
                    
    #                 raw_protos[class_index] = raw_proto
    #                 class_counts[class_index] = count
    #                 class_variances[class_index] = avg_variance
                
    #             # 第二步：计算每个类别的质量分数
    #             quality_scores = {}
    #             for class_index in class_list:
    #                 count = class_counts[class_index]
    #                 variance = class_variances[class_index]
                    
    #                 # 样本数量分数（0-1）
    #                 count_score = min(1.0, count / (2 * min_samples)) if count > 0 else 0.0
                    
    #                 # 方差分数（0-1），方差越小分数越高
    #                 var_score = max(0.0, 1.0 - variance / var_threshold)
                    
    #                 # 综合质量分数（几何平均）
    #                 quality_scores[class_index] = math.sqrt(count_score * var_score)
                
    #             # 计算质量分数分布统计
    #             quality_values = list(quality_scores.values())
    #             mean_quality = sum(quality_values) / len(quality_values) if quality_values else 0
    #             max_quality = max(quality_values) if quality_values else 0
    #             min_quality = min(quality_values) if quality_values else 0
                
    #             # 第三步：根据质量分数调整原型
    #             adjusted_protos = {}  # 存储调整后的原型
    #             for class_index in class_list:
    #                 raw_proto = raw_protos[class_index]
    #                 quality = quality_scores[class_index]
                    
    #                 if quality > 0.7:
    #                     # 高质量原型：直接使用
    #                     adjusted_proto = raw_proto
    #                 elif quality > 0.4:
    #                     # 中等质量：向全局中心轻微收缩
    #                     shrink_factor = 0.3 * (1 - quality)  # 质量越低，收缩越多
    #                     adjusted_proto = (1 - shrink_factor) * raw_proto + shrink_factor * global_center
    #                 else:
    #                     # 低质量：向全局中心显著收缩
    #                     shrink_factor = 0.7 * (1 - quality)  # 质量越低，收缩越多
    #                     adjusted_proto = (1 - shrink_factor) * raw_proto + shrink_factor * global_center
                        
    #                     # 额外：向高质量原型靠拢（如果存在）
    #                     if max_quality > 0.7:
    #                         # 找到最相似的高质量原型
    #                         best_similarity = -1
    #                         best_proto = None
    #                         for other_class, other_quality in quality_scores.items():
    #                             if other_quality > 0.7 and other_class != class_index:
    #                                 other_proto = raw_protos[other_class]  # 使用之前计算的原始原型
    #                                 similarity = F.cosine_similarity(
    #                                     raw_proto.unsqueeze(0), 
    #                                     other_proto.unsqueeze(0)
    #                                 ).item()
    #                                 if similarity > best_similarity:
    #                                     best_similarity = similarity
    #                                     best_proto = other_proto
                            
    #                         if best_proto is not None and best_similarity > 0.4:
    #                             # 向高质量原型靠拢
    #                             adjust_factor = 0.4 * (1 - quality)
    #                             adjusted_proto = (1 - adjust_factor) * adjusted_proto + adjust_factor * best_proto
                    
    #                 adjusted_protos[class_index] = adjusted_proto
                
    #             # 第四步：存储原型并更新FC层
    #             for class_index in class_list:
    #                 adjusted_proto = adjusted_protos[class_index]
    #                 # 存储调整后的原型
    #                 self.class_proto = torch.cat((self.class_proto, adjusted_proto.unsqueeze(0).to(self._device)), dim=0)
                    
    #                 # 更新FC层权重
    #                 if self.use_init_ptm:
    #                     model.fc.weight.data[class_index, (index+1)*self._network.out_dim:(index+2)*self._network.out_dim] = adjusted_proto
    #                 else:
    #                     model.fc.weight.data[class_index, index*self._network.out_dim:(index+1)*self._network.out_dim] = adjusted_proto
        
    #     return
    
    # def replace_fc(self, train_loader):
    #     """
    #     使用细化的原型计算方法，考虑样本数量和数据分布特性
    #     """
    #     model = self._network
    #     model = model.eval()
        
    #     with torch.no_grad():
    #         if self.use_init_ptm:
    #             start_idx = -1
    #         else:
    #             start_idx = 0
                
    #         for index in range(start_idx, self._cur_task + 1):
    #             if self.moni_adam and index > self.adapter_num - 1:
    #                 break
    #             elif self.use_diagonal and index != -1 and index != self._cur_task:
    #                 continue
                    
    #             embedding_list, label_list = [], []
                
    #             # 收集特征和标签
    #             for i, batch in enumerate(train_loader):
    #                 (_, data, label) = batch
    #                 data = data.to(self._device)
    #                 label = label.to(self._device)
    #                 embedding = model.backbone.forward_proto(data, adapt_index=index)
    #                 embedding_list.append(embedding.cpu())
    #                 label_list.append(label.cpu())
                    
    #             embedding_list = torch.cat(embedding_list, dim=0)
    #             label_list = torch.cat(label_list, dim=0)
    #             class_list = np.unique(self.train_dataset_for_protonet.labels)
                
    #             # 计算全局统计信息用于标准化
    #             all_sample_counts = []
    #             all_variances = []
                
    #             for class_index in class_list:
    #                 data_index = (label_list == class_index).nonzero().squeeze(-1)
    #                 all_sample_counts.append(len(data_index))
    #                 if len(data_index) > 1:
    #                     class_embeddings = embedding_list[data_index]
    #                     variance = torch.var(class_embeddings, dim=0).mean().item()
    #                     all_variances.append(variance)
                
    #             # 计算统计量用于标准化
    #             max_count = max(all_sample_counts)
    #             mean_variance = np.mean(all_variances) if all_variances else 1.0
                
    #             # 为每个类计算细化的原型
    #             for class_index in class_list:
    #                 data_index = (label_list == class_index).nonzero().squeeze(-1)
    #                 embedding = embedding_list[data_index]
    #                 sample_count = len(data_index)
                    
    #                 # 方案1: 基于样本数量的置信度调整
    #                 proto_refined = self._compute_confidence_weighted_proto(
    #                     embedding, sample_count, max_count
    #                 )
                    
    #                 # 方案2: 结合方差信息的调整
    #                 if sample_count > 1:
    #                     proto_refined = self._compute_variance_aware_proto(
    #                         embedding, sample_count, mean_variance
    #                     )
                    
    #                 # 方案3: 综合多种因素
    #                 proto_refined = self._compute_comprehensive_proto(
    #                     embedding, sample_count, max_count, mean_variance
    #                 )
                    
    #                 self.class_proto = torch.cat((self.class_proto, proto_refined.unsqueeze(0).to(self._device)), dim=0)
                    
    #                 # 更新分类器权重
    #                 if self.use_init_ptm:
    #                     model.fc.weight.data[class_index, (index+1)*self._network.out_dim:(index+2)*self._network.out_dim] = proto_refined
    #                 else:
    #                     model.fc.weight.data[class_index, index*self._network.out_dim:(index+1)*self._network.out_dim] = proto_refined
        
    #     return
    
    def _compute_confidence_weighted_proto(self, embeddings, sample_count, max_count):
        """
        方案1: 基于样本数量的置信度加权
        对于样本少的类，向全局中心或预训练知识回退
        """
        # 计算基础原型
        base_proto = embeddings.mean(0)
        
        # 置信度基于样本数量
        confidence = min(1.0, sample_count / (max_count * 0.3))  # 可调节阈值
        
        # 如果置信度低，向零向量（或全局均值）回退
        if hasattr(self, 'global_mean_embedding'):
            fallback_proto = self.global_mean_embedding
        else:
            fallback_proto = torch.zeros_like(base_proto)
        
        # 加权组合
        refined_proto = confidence * base_proto + (1 - confidence) * fallback_proto
        
        return refined_proto

    def _compute_variance_aware_proto(self, embeddings, sample_count, mean_variance):
        """
        方案2: 考虑类内方差的原型计算
        方差大的类可能需要更保守的原型
        """
        base_proto = embeddings.mean(0)
        
        if sample_count > 1:
            # 计算类内方差
            class_variance = torch.var(embeddings, dim=0).mean().item()
            
            # 方差相对于全局的比率
            variance_ratio = class_variance / (mean_variance + 1e-8)
            
            # 高方差时更保守，低方差时更激进
            shrinkage_factor = min(0.9, 1.0 / (1.0 + variance_ratio))
            
            # 向均值收缩
            refined_proto = shrinkage_factor * base_proto
        else:
            refined_proto = base_proto
        
        return refined_proto
    
    # def _compute_comprehensive_proto(self, embeddings, sample_count, max_count, mean_variance):
    #     """
    #     方案3: 综合考虑多种因素的原型计算
    #     """
    #     base_proto = embeddings.mean(0)
        
    #     # 样本数量置信度
    #     count_confidence = np.tanh(sample_count / 10.0)  # 平滑的置信度函数
        
    #     # 方差稳定性
    #     if sample_count > 1:
    #         class_variance = torch.var(embeddings, dim=0).mean().item()
    #         variance_stability = 1.0 / (1.0 + class_variance / mean_variance)
    #     else:
    #         variance_stability = 0.5  # 单样本情况下的默认值
        
    #     # 综合权重
    #     overall_confidence = 0.7 * count_confidence + 0.3 * variance_stability
        
    #     # 自适应正则化
    #     regularization_strength = 1.0 - overall_confidence
        
    #     # L2正则化（向零收缩）
    #     refined_proto = base_proto * (1.0 - regularization_strength * 0.1)
        
    #     # 可选：添加噪声以增强泛化能力（对于低置信度的类）
    #     if overall_confidence < 0.5 and sample_count < 5:
    #         noise_scale = (1.0 - overall_confidence) * 0.01
    #         noise = torch.randn_like(refined_proto) * noise_scale
    #         refined_proto += noise
        
    #     return refined_proto
    
    def _compute_comprehensive_proto(self, embeddings, sample_count, max_count, mean_variance, 
                                class_index=None, all_class_protos=None):
        """
        方案3: 综合考虑多种因素的原型计算 - 增强类间区分性版本
        """
        base_proto = embeddings.mean(0)
        
        # 样本数量置信度
        count_confidence = np.tanh(sample_count / 10.0)
        
        # 方差稳定性
        if sample_count > 1:
            class_variance = torch.var(embeddings, dim=0).mean().item()
            variance_stability = 1.0 / (1.0 + class_variance / mean_variance)
        else:
            variance_stability = 0.5
        
        # 综合权重
        overall_confidence = 0.7 * count_confidence + 0.3 * variance_stability
        
        # === 新增：增强类间区分性 ===
        if all_class_protos is not None and len(all_class_protos) > 0:
            # 计算与其他类原型的最小距离
            min_inter_distance = self._compute_min_inter_class_distance(base_proto, all_class_protos)
            
            # 如果类间距离过小，增强原型的区分性
            distance_threshold = 0.1  # 可调节
            if min_inter_distance < distance_threshold:
                # 计算推离方向
                repulsion_vector = self._compute_repulsion_vector(base_proto, all_class_protos)
                # 应用推离，但要考虑置信度
                repulsion_strength = (1.0 - min_inter_distance / distance_threshold) * overall_confidence * 0.1
                base_proto = base_proto + repulsion_strength * repulsion_vector
        
        # 自适应正则化 - 降低正则化强度以保持区分性
        regularization_strength = (1.0 - overall_confidence) * 0.05  # 从0.1降到0.05
        
        # 更温和的正则化
        refined_proto = base_proto * (1.0 - regularization_strength)
        
        # 对于极少样本的情况，使用对比学习思想
        if sample_count < 3 and all_class_protos is not None:
            refined_proto = self._apply_contrastive_adjustment(
                refined_proto, all_class_protos, overall_confidence
            )
        
        return refined_proto

    def _update_global_statistics(self, train_loader):
        """
        预计算全局统计信息
        """
        all_embeddings = []
        
        with torch.no_grad():
            for batch in train_loader:
                (_, data, _) = batch
                data = data.to(self._device)
                embedding = self._network.backbone.forward_proto(data, adapt_index=-1)
                all_embeddings.append(embedding.cpu())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        self.global_mean_embedding = all_embeddings.mean(0)
        self.global_std_embedding = all_embeddings.std(0)

    # 新增辅助方法
    def _compute_min_inter_class_distance(self, current_proto, existing_protos):
        """
        计算当前原型与已存在原型的最小距离
        """
        if len(existing_protos) == 0:
            return float('inf')
        
        distances = []
        for proto in existing_protos:
            # 使用余弦距离（与最终分类保持一致）
            cosine_sim = torch.cosine_similarity(current_proto.unsqueeze(0), proto.unsqueeze(0))
            distance = 1.0 - cosine_sim.item()
            distances.append(distance)
        
        return min(distances)

    def _compute_repulsion_vector(self, current_proto, existing_protos):
        """
        计算推离向量，使当前原型远离过近的其他原型
        """
        repulsion_vector = torch.zeros_like(current_proto)
        
        for proto in existing_protos:
            # 计算从其他原型指向当前原型的方向
            direction = current_proto - proto
            distance = torch.norm(direction) + 1e-8  # 避免除零
            
            # 标准化方向向量，距离越近权重越大
            weight = 1.0 / (distance + 1e-8)
            repulsion_vector += weight * direction / distance
        
        # 标准化总的推离向量
        if torch.norm(repulsion_vector) > 1e-8:
            repulsion_vector = repulsion_vector / torch.norm(repulsion_vector)
        
        return repulsion_vector

    def _apply_contrastive_adjustment(self, proto, existing_protos, confidence):
        """
        基于对比学习的原型调整
        """
        if len(existing_protos) == 0:
            return proto
        
        # 计算与所有已存在原型的相似度
        similarities = []
        for existing_proto in existing_protos:
            sim = torch.cosine_similarity(proto.unsqueeze(0), existing_proto.unsqueeze(0))
            similarities.append(sim.item())
        
        # 如果最大相似度过高，进行调整
        max_similarity = max(similarities)
        if max_similarity > 0.9:  # 阈值可调
            # 找到最相似的原型
            most_similar_idx = similarities.index(max_similarity)
            most_similar_proto = existing_protos[most_similar_idx]
            
            # 计算垂直方向的调整
            diff = proto - most_similar_proto
            adjustment_strength = (max_similarity - 0.9) * confidence * 0.5
            
            # 在垂直于最相似原型的方向上调整
            orthogonal_direction = diff - torch.dot(diff, most_similar_proto) * most_similar_proto
            if torch.norm(orthogonal_direction) > 1e-8:
                orthogonal_direction = orthogonal_direction / torch.norm(orthogonal_direction)
                proto = proto + adjustment_strength * orthogonal_direction
        
        return proto

    # def replace_fc(self, train_loader):
    #     """
    #     使用细化的原型计算方法，考虑样本数量和数据分布特性
    #     """
    #     model = self._network
    #     model = model.eval()
        
    #     with torch.no_grad():
    #         if self.use_init_ptm:
    #             start_idx = -1
    #         else:
    #             start_idx = 0
                
    #         for index in range(start_idx, self._cur_task + 1):
    #             if self.moni_adam and index > self.adapter_num - 1:
    #                 break
    #             elif self.use_diagonal and index != -1 and index != self._cur_task:
    #                 continue
                    
    #             embedding_list, label_list = [], []
                
    #             # 收集特征和标签
    #             for i, batch in enumerate(train_loader):
    #                 (_, data, label) = batch
    #                 data = data.to(self._device)
    #                 label = label.to(self._device)
    #                 embedding = model.backbone.forward_proto(data, adapt_index=index)
    #                 embedding_list.append(embedding.cpu())
    #                 label_list.append(label.cpu())
                    
    #             embedding_list = torch.cat(embedding_list, dim=0)
    #             label_list = torch.cat(label_list, dim=0)
    #             class_list = np.unique(self.train_dataset_for_protonet.labels)
                
    #             # 计算全局统计信息用于标准化
    #             all_sample_counts = []
    #             all_variances = []
                
    #             for class_index in class_list:
    #                 data_index = (label_list == class_index).nonzero().squeeze(-1)
    #                 all_sample_counts.append(len(data_index))
    #                 if len(data_index) > 1:
    #                     class_embeddings = embedding_list[data_index]
    #                     variance = torch.var(class_embeddings, dim=0).mean().item()
    #                     all_variances.append(variance)
                
    #             # 计算统计量用于标准化
    #             max_count = max(all_sample_counts)
    #             mean_variance = np.mean(all_variances) if all_variances else 1.0
                
    #             # 为每个类计算细化的原型
    #             for class_index in class_list:
    #                 data_index = (label_list == class_index).nonzero().squeeze(-1)
    #                 embedding = embedding_list[data_index]
    #                 sample_count = len(data_index)
                    
    #                 # 方案1: 基于样本数量的置信度调整
    #                 proto_refined = self._compute_confidence_weighted_proto(
    #                     embedding, sample_count, max_count
    #                 )
                    
    #                 # 方案2: 结合方差信息的调整
    #                 if sample_count > 1:
    #                     proto_refined = self._compute_variance_aware_proto(
    #                         embedding, sample_count, mean_variance
    #                     )
                    
    #                 # 方案3: 综合多种因素
    #                 proto_refined = self._compute_comprehensive_proto(
    #                     embedding, sample_count, max_count, mean_variance
    #                 )
                    
    #                 self.class_proto = torch.cat((self.class_proto, proto_refined.unsqueeze(0).to(self._device)), dim=0)
                    
    #                 # 更新分类器权重
    #                 if self.use_init_ptm:
    #                     model.fc.weight.data[class_index, (index+1)*self._network.out_dim:(index+2)*self._network.out_dim] = proto_refined
    #                 else:
    #                     model.fc.weight.data[class_index, index*self._network.out_dim:(index+1)*self._network.out_dim] = proto_refined
        
    #     return
    
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train", )
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        
        self.test_dataset_for_cur_task = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="test", mode="test", )
        self.test_loader_for_cur_task = DataLoader(self.test_dataset_for_cur_task, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        self.train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(self.train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader_for_cur_task)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.add_fc()
        self.replace_fc(self.train_loader_for_protonet)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self.task_increments:
            self.inc = self.task_increments[self._cur_task]
        
        if self._cur_task == 0 or self.init_cls == self.inc:
            optimizer = self.get_optimizer(lr=self.args["init_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["init_epochs"])
        else:
            # for base 0 setting, the later_lr and later_epochs are not used
            # for base N setting, the later_lr and later_epochs are used
            if "later_lr" not in self.args or self.args["later_lr"] == 0:
                self.args["later_lr"] = self.args["init_lr"]
            if "later_epochs" not in self.args or self.args["later_epochs"] == 0:
                self.args["later_epochs"] = self.args["init_epochs"]

            optimizer = self.get_optimizer(lr=self.args["later_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["later_epochs"])

        self._init_train(train_loader, test_loader, optimizer, scheduler)
        


    def get_optimizer(self, lr):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr,
                weight_decay=self.weight_decay
            )

        return optimizer

    def get_scheduler(self, optimizer, epoch):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    # def _init_train(self, train_loader, test_loader, optimizer, scheduler):
    #     if self.moni_adam:
    #         if self._cur_task > self.adapter_num - 1:
    #             return
    #     if self.task_increments:
    #         self.inc = self.task_increments[self._cur_task]
    #     if self._cur_task == 0 or self.init_cls == self.inc:
    #         epochs = self.args['init_epochs']
    #     else:
    #         epochs = self.args['later_epochs']

    #     if not self._network.backbone.msa_adapt:
    #         for name, param in self._network.backbone.cur_adapter[0].named_parameters():
    #             print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
    #     else:
    #         for name, param in self._network.backbone.cur_adapter[0][1].named_parameters():
    #             print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
    #         for name, param in self._network.backbone.cur_adapter[-1][1].named_parameters():
    #             print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
                
    #     prog_bar = tqdm(range(epochs))
    #     best_acc, best_loss = 0.0, 100000.0  
    #     for _, epoch in enumerate(prog_bar):
    #         self._network.train()

    #         losses = 0.0
    #         correct, total = 0, 0
    #         for i, (_, inputs, targets) in enumerate(train_loader):
    #             inputs, targets = inputs.to(self._device), targets.to(self._device)
    #             aux_targets = targets.clone()

    #             aux_targets = torch.where(
    #                 aux_targets - self._known_classes >= 0,
    #                 aux_targets - self._known_classes,
    #                 -1,
    #             )
    #             output = self._network(inputs, test=False)

    #             logits = output["logits"]
    #             loss = F.cross_entropy(logits, aux_targets)
    #             reg_loss = sum([lora.regularization_loss() for lora in self._network.backbone.cur_adapter])
    #             loss = loss + reg_loss
                
    #             # 反向传播前添加梯度裁剪
    #             for lora in self._network.backbone.cur_adapter:
    #                 lora.clip_gradients(max_norm=1.0)
                
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             losses += loss.item()
    #             _, preds = torch.max(logits, dim=1)

    #             correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()
    #             total += len(aux_targets)
        
    #         if scheduler:
    #             scheduler.step()
    #         train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    #         if losses < best_loss:
    #             best_loss = losses
    #             best_acc = train_acc
    #             save_path = self.args["logs_name"] + f"/best_model_task_{self._cur_task}.pth"
    #             torch.save({
    #                 'state_dict': self._network.state_dict(),
    #                 'lora_ranks': [lora.current_rank for lora in self._network.backbone.cur_adapter],  # 保存当前秩
    #                 'task_id': self._cur_task,
    #                 'best_acc': best_acc
    #             }, save_path)

    #         info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Best_accy {:.2f}".format(
    #                 self._cur_task,
    #                 epoch + 1,
    #                 epochs,
    #                 losses / len(train_loader),
    #                 train_acc,
    #                 best_acc
    #             )
    #         prog_bar.set_description(info)

    #         logging.info(info)
    #     checkpoint = torch.load(save_path)    
    #     self._network.load_state_dict(checkpoint['state_dict'])
    #     for lora, saved_rank in zip(self._network.backbone.cur_adapter, checkpoint['lora_ranks']):
    #         lora.current_rank = saved_rank

    def _init_train(self, train_loader, test_loader, optimizer, scheduler): 
        if self.moni_adam: 
            if self._cur_task > self.adapter_num - 1: 
                return 
        
        if self.task_increments: 
            self.inc = self.task_increments[self._cur_task] 
        
        if self._cur_task == 0 or self.init_cls == self.inc: 
            epochs = self.args['init_epochs'] 
        else: 
            epochs = self.args['later_epochs'] 

        # # 修复：更安全的参数检查和LoRA组件访问
        # self._check_lora_parameters()
        
        # 在训练开始前初始化LoRA的有效秩
        self._initialize_lora_ranks()
        
        prog_bar = tqdm(range(epochs)) 
        best_acc, best_loss = 0.0, float('inf')
        
        # 添加早停机制相关变量
        patience = getattr(self.args, 'patience', 15)
        patience_counter = 0
        self.cur_class_proto = torch.zeros((self.inc, 768)).to(self._device)
        for epoch in range(epochs): 
            self._network.train() 
            
            # 训练一个epoch
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, epoch)
            
            # # 验证
            # if test_loader is not None:
            #     val_acc, val_loss = self._validate_epoch(test_loader)
            # else:
            #     val_acc, val_loss = train_acc, train_loss
            
            val_acc, val_loss = train_acc, train_loss
                
            # 学习率调整
            if scheduler: 
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step() 
            
            # 保存最佳模型
            is_best = val_loss < best_loss
            if is_best and epoch > 5:
                best_loss = val_loss
                best_acc = val_acc
                patience_counter = 0
                self._save_best_model(epoch, val_acc, val_loss, optimizer)
            else:
                patience_counter += 1
            
            # 更新进度条信息
            info = ("Task {}, Epoch {}/{} => Train_Loss {:.3f}, Train_Acc {:.2f}%, "
                    "Val_Loss {:.3f}, Val_Acc {:.2f}%, Best_Acc {:.2f}%, LR {:.2e}").format(
                    self._cur_task, 
                    epoch + 1, 
                    epochs, 
                    train_loss, 
                    train_acc,
                    val_loss,
                    val_acc,
                    best_acc,
                    optimizer.param_groups[0]['lr']
                ) 
            prog_bar.set_description(info)
            prog_bar.update(1)
            logging.info(info)
            
            # 早停检查
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            # # 定期进行LoRA剪枝（可选）
            # if (epoch + 1) % getattr(self.args, 'prune_interval', 20) == 0:
            #     self._prune_lora_adapters(epoch)
        
        prog_bar.close()
        
        # 加载最佳模型
        self._load_best_model()
        
        # lora_adapters = self._get_lora_adapters()
        # for lora in lora_adapters:
        #     # if hasattr(lora, 'commit_current_as_base'):
        #     #     lora.commit_current_as_base()
        #     if hasattr(lora, 'prune_parameters'):
        #         lora.prune_parameters()
        
        # 训练结束后的LoRA统计
        self._log_lora_statistics()

    def _check_lora_parameters(self):
        """检查LoRA参数的梯度状态"""
        lora_adapters = self._get_lora_adapters()
        
        for i, lora in enumerate(lora_adapters):
            print(f"\n=== LoRA Adapter {i} Parameters ===")
            for name, param in lora.named_parameters():
                print(f"Parameter: {name}, Shape: {param.shape}, Requires Gradient: {param.requires_grad}")
                if hasattr(lora, 'get_effective_rank'):
                    print(f"Effective Rank: {lora.get_effective_rank()}")

    def _get_lora_adapters(self):
        """获取所有LoRA适配器"""
        adapters = []
        
        if not getattr(self._network.backbone, 'msa_adapt', False):
            # 普通情况：每层一个adapter
            if hasattr(self._network.backbone, 'cur_adapter'):
                for adapter in self._network.backbone.cur_adapter:
                    if hasattr(adapter, 'regularization_loss'):  # 确认是LoRA adapter
                        adapters.append(adapter)
        else:
            # MSA适配情况：需要特殊处理
            if hasattr(self._network.backbone, 'cur_adapter'):
                for layer_adapters in self._network.backbone.cur_adapter:
                    if isinstance(layer_adapters, (list, tuple, nn.ModuleList)):
                        for adapter in layer_adapters:
                            if hasattr(adapter, 'regularization_loss'):
                                adapters.append(adapter)
                    elif hasattr(layer_adapters, 'regularization_loss'):
                        adapters.append(layer_adapters)
        
        return adapters

    def _initialize_lora_ranks(self):
        """初始化LoRA的有效秩"""
        lora_adapters = self._get_lora_adapters()
        for lora in lora_adapters:
            if hasattr(lora, 'get_effective_rank'):
                lora.current_rank = lora.get_effective_rank()

    def _train_epoch(self, train_loader, optimizer, epoch_index):
        """训练一个epoch"""
        total_loss = 0.0
        correct, total = 0, 0
        lora_adapters = self._get_lora_adapters()
        cur_class_proto = torch.zeros((self.inc, 768)).to(self._device)
        cur_class_nums = torch.ones((self.inc)).to(self._device)
        for i, (_, inputs, targets) in enumerate(train_loader): 
            inputs, targets = inputs.to(self._device), targets.to(self._device) 
            
            # 修复：目标标签处理
            aux_targets = targets.clone() 
            aux_targets = torch.where( 
                aux_targets - self._known_classes >= 0, 
                aux_targets - self._known_classes, 
                -1, 
            )
            
            # 过滤掉无效标签
            valid_mask = aux_targets >= 0
            if not valid_mask.any():
                continue
                
            inputs = inputs[valid_mask]
            aux_targets = aux_targets[valid_mask]
            
            # 前向传播
            optimizer.zero_grad()
            output = self._network(inputs, test=False) 
            logits = output["logits"] 
            
            # 计算损失
            ce_loss = F.cross_entropy(logits, aux_targets)
            # 计算anti-prototype loss
            if epoch_index > 5:
                anti_prototype_loss = self.anti_prototype_loss(features = output['features'], labels = aux_targets, cur_proto = self.cur_class_proto, prev_proto = self.class_proto)
            else:
                anti_prototype_loss = self.anti_prototype_loss(output['features'], None, None, self.class_proto)
            # 添加LoRA正则化损失
            reg_loss = torch.tensor(0.0, device=self._device)
            for lora in lora_adapters:
                if hasattr(lora, 'regularization_loss'):
                    reg_loss += lora.regularization_loss()
            
            total_loss_batch = ce_loss + 1 * anti_prototype_loss[0]+ reg_loss#+ reg_loss#+ reg_loss + 1 * anti_prototype_loss[0] # ce_loss + 
            
            # 反向传播
            total_loss_batch.backward()
            
            # 梯度裁剪（在optimizer.step()之前）
            # for lora in lora_adapters:
            #     if hasattr(lora, 'clip_gradients'):
            #         lora.clip_gradients(max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                for c in aux_targets.unique():
                    mask = (aux_targets == c)
                    if mask.any():
                        cur_class_proto[c] += output['features'][mask].sum(dim=0)
                        cur_class_nums[c] += mask.sum().item()
            # 统计
            total_loss += total_loss_batch.item()
            _, preds = torch.max(logits, dim=1) 
            correct += preds.eq(aux_targets).cpu().sum().item()
            total += len(aux_targets)
        
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_acc = 100.0 * correct / total if total > 0 else 0.0
        self.cur_class_proto = cur_class_proto/cur_class_nums.unsqueeze(-1)
        # if epoch_index % 25 == 0:
        #     for lora in lora_adapters:
        #         if hasattr(lora, 'prune_parameters'):
        #             lora.prune_parameters()
            # if hasattr(lora, 'prune_parameters'):
            #     lora.prune_parameters()
        return avg_loss, avg_acc

    def _save_best_model(self, epoch, acc, loss,optimizer):
        """保存最佳模型"""
        lora_adapters = self._get_lora_adapters()
        lora_stats = []
        
        for lora in lora_adapters:
            stats = {
                'current_rank': getattr(lora, 'current_rank', 0),
                'effective_rank': lora.get_effective_rank() if hasattr(lora, 'get_effective_rank') else 0,
                'max_rank': getattr(lora, 'max_rank', 0)
            }
            lora_stats.append(stats)
        
        save_path = self.args["logs_name"] + f"/best_model_task_{self._cur_task}.pth"
        logging.info(f"Save best model to {save_path}")
        torch.save({
            'state_dict': self._network.state_dict(),
            'lora_stats': lora_stats,
            'task_id': self._cur_task,
            'epoch': epoch,
            'best_acc': acc,
            'best_loss': loss,
            'optimizer_state': optimizer.state_dict() if hasattr(self, 'optimizer') else None
        }, save_path)

    def _load_best_model(self):
        """加载最佳模型"""
        save_path = self.args["logs_name"] + f"/best_model_task_{self._cur_task}.pth"
        
        try:
            checkpoint = torch.load(save_path, map_location=self._device)
            self._network.load_state_dict(checkpoint['state_dict'])
            
            # 恢复LoRA统计信息
            if 'lora_stats' in checkpoint:
                lora_adapters = self._get_lora_adapters()
                for lora, stats in zip(lora_adapters, checkpoint['lora_stats']):
                    lora.current_rank = stats.get('current_rank', lora.current_rank)
                    
            logging.info(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
        except FileNotFoundError:
            logging.warning(f"Best model file not found: {save_path}")

    def _prune_lora_adapters(self, epoch):
        """定期进行LoRA剪枝"""
        if not getattr(self.args, 'enable_pruning', True):
            return
            
        lora_adapters = self._get_lora_adapters()
        for i, lora in enumerate(lora_adapters):
            if hasattr(lora, 'prune_parameters'):
                old_rank = getattr(lora, 'current_rank', 0)
                lora.prune_parameters()
                
                new_rank = getattr(lora, 'current_rank', 0)
                if old_rank != new_rank:
                    logging.info(f"LoRA {i} pruned: {old_rank} -> {new_rank} at epoch {epoch + 1}")

    def _log_lora_statistics(self):
        """记录LoRA统计信息"""
        lora_adapters = self._get_lora_adapters()
        
        total_params = 0
        active_params = 0
        
        for i, lora in enumerate(lora_adapters):
            if hasattr(lora, 'get_effective_rank'):
                effective_rank = lora.get_active_components()[-1] # lora.get_effective_rank()
                max_rank = getattr(lora, 'max_rank', 0)
                
                lora_params = effective_rank * (lora.n_embd + lora.n_embd) if hasattr(lora, 'n_embd') else 0
                max_params = max_rank * (lora.n_embd + lora.n_embd) if hasattr(lora, 'n_embd') else 0
                
                total_params += max_params
                active_params += lora_params
                
                logging.info(f"LoRA {i}: Effective Rank {effective_rank}/{max_rank}, "
                            f"Params {lora_params}/{max_params}")
        
        if total_params > 0:
            compression_ratio = active_params / total_params
            logging.info(f"Overall LoRA Compression: {compression_ratio:.3f} "
                        f"({active_params}/{total_params} parameters)")

    def _validate_epoch(self, test_loader):
        """验证一个epoch"""
        self._network.eval()
        total_loss = 0.0
        correct, total = 0, 0
        
        with torch.no_grad():
            for _, inputs, targets in test_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                # 处理目标标签
                aux_targets = targets.clone()
                aux_targets = torch.where(
                    aux_targets - self._known_classes >= 0,
                    aux_targets - self._known_classes,
                    -1,
                )
                
                valid_mask = aux_targets >= 0
                if not valid_mask.any():
                    continue
                    
                inputs = inputs[valid_mask]
                aux_targets = aux_targets[valid_mask]
                
                output = self._network(inputs, test=False)
                logits = output["logits"]
                
                loss = F.cross_entropy(logits, aux_targets)
                total_loss += loss.item()
                
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(aux_targets).cpu().sum().item()
                total += len(aux_targets)
        
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        avg_acc = 100.0 * correct / total if total > 0 else 0.0
        
        return avg_acc, avg_loss
    
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.forward(inputs, test=True)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        calc_task_acc = True
        if self.task_increments:
            self.inc = self.task_increments[self._cur_task]
        if calc_task_acc:
            task_correct, task_acc, total = 0, 0, 0

        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                outputs = self._network.forward(inputs, test=True)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

            # calculate the accuracy by using task_id
            if calc_task_acc:
                task_ids = (targets - self.init_cls) // self.inc + 1
                task_logits = torch.zeros(outputs.shape).to(self._device)
                for i, task_id in enumerate(task_ids):
                    if task_id == 0:
                        start_cls = 0
                        end_cls = self.init_cls
                    else:
                        start_cls = self.init_cls + (task_id-1)*self.inc
                        end_cls = self.init_cls + task_id*self.inc
                    task_logits[i, start_cls:end_cls] += outputs[i, start_cls:end_cls]
                # calculate the accuracy of task_id
                pred_task_ids = (torch.max(outputs, dim=1)[1] - self.init_cls) // self.inc + 1
                task_correct += (pred_task_ids.cpu() == task_ids).sum()

                pred_task_y = torch.max(task_logits, dim=1)[1]
                task_acc += (pred_task_y.cpu() == targets).sum()
                total += len(targets)

        if calc_task_acc:
            logging.info("Task correct: {}".format(tensor2numpy(task_correct) * 100 / total))
            logging.info("Task acc: {}".format(tensor2numpy(task_acc) * 100 / total))

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
