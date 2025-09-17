import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report

# class DomainContrastiveLoss(nn.Module):
#     def __init__(self, margin=0.5, alpha=1.0, beta=1.0, gamma=0.3, temperature=0.07):
#         """
#         改进的域对比损失 (支持类别不平衡 + 跨域正对齐)
#         Args:
#             margin: 负约束的最小余弦距离
#             alpha: 正对齐损失权重
#             beta: 负分离损失权重
#             gamma: 类内紧凑性损失权重
#             temperature: 相似度分布锐化系数
#         """
#         super().__init__()
#         self.margin = margin
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.temperature = temperature

#     def forward(self, features, labels=None, cur_proto=None, prev_proto=None, all_proto_labels=None, class_weights=None):
#         """
#         Args:
#             features: [B, D] 当前 batch 特征
#             labels:   [B] 当前 batch 标签
#             cur_proto: [C1, D] 当前任务类别原型
#             prev_proto: [C2, D] 历史任务类别原型
#             all_proto_labels: [C1 + C2] 所有原型对应的类别标签
#             class_weights: [num_classes] 每个类别的权重 (解决类别不平衡)
#         """
#         B, D = features.size()
#         f_norm = F.normalize(features, p=2, dim=1)

#         loss_dict = {"loss_pos": 0.0, "loss_neg": 0.0, "loss_intra": 0.0}
#         total_loss = torch.tensor(0.0, device=features.device)

#         # ==================================================
#         # 1. 组织所有原型 (当前 + 历史)
#         # ==================================================
#         all_proto = []
#         if cur_proto is not None:
#             all_proto.append(cur_proto)
#         if prev_proto is not None and prev_proto.size(0) > 0:
#             all_proto.append(prev_proto)

#         if len(all_proto) > 0:
#             all_proto = torch.cat(all_proto, dim=0)  # [C_all, D]
#             all_proto_norm = F.normalize(all_proto, p=2, dim=1)  # 单位化
#         else:
#             all_proto, all_proto_norm = None, None

#         # ==================================================
#         # 2. 正对齐损失 (跨域同类原型对齐)
#         # ==================================================
#         if all_proto is not None and labels is not None and all_proto_labels is not None:
#             pos_losses = []
#             for i in range(B):
#                 # 找到该样本对应的所有同类原型
#                 same_class_mask = (all_proto_labels == labels[i])
#                 same_class_proto = all_proto_norm[same_class_mask]

#                 if same_class_proto.size(0) > 0:
#                     # 与同类所有原型计算相似度
#                     pos_sim = torch.matmul(f_norm[i], same_class_proto.t())
#                     pos_loss = (1 - pos_sim).clamp(min=0).mean()

#                     # 类别权重 (如果提供)
#                     if class_weights is not None:
#                         pos_loss = pos_loss * class_weights[labels[i]]

#                     pos_losses.append(pos_loss)

#             if len(pos_losses) > 0:
#                 loss_pos = torch.stack(pos_losses).mean()
#                 total_loss += self.alpha * loss_pos
#                 loss_dict["loss_pos"] = loss_pos.item()

#         # ==================================================
#         # 3. 负分离损失 (与异类原型分离)
#         # ==================================================
#         if all_proto is not None and labels is not None and all_proto_labels is not None:
#             neg_losses = []
#             for i in range(B):
#                 # 找到该样本对应的所有异类原型
#                 diff_class_mask = (all_proto_labels != labels[i])
#                 diff_class_proto = all_proto_norm[diff_class_mask]

#                 if diff_class_proto.size(0) > 0:
#                     # 相似度 (加温度)
#                     neg_sim = torch.matmul(f_norm[i], diff_class_proto.t()) / self.temperature
#                     neg_dist = 1 - neg_sim  # 余弦距离

#                     # 选择最难负样本 (最近的异类原型)
#                     hard_neg_dist, _ = neg_dist.min(dim=0)
#                     neg_loss = F.relu(self.margin - hard_neg_dist).mean()

#                     # 类别权重
#                     if class_weights is not None:
#                         neg_loss = neg_loss * class_weights[labels[i]]

#                     neg_losses.append(neg_loss)

#             if len(neg_losses) > 0:
#                 loss_neg = torch.stack(neg_losses).mean()
#                 total_loss += self.beta * loss_neg
#                 loss_dict["loss_neg"] = loss_neg.item()

#         # ==================================================
#         # 4. 类内紧凑性损失 (batch 内同类样本保持紧凑)
#         # ==================================================
#         if labels is not None and B > 1:
#             same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
#             same_class.fill_diagonal_(False)

#             if same_class.sum() > 0:
#                 intra_sim = torch.mm(f_norm, f_norm.t())
#                 intra_loss = (1 - intra_sim[same_class]).mean()

#                 total_loss += self.gamma * intra_loss
#                 loss_dict["loss_intra"] = intra_loss.item()

#         return total_loss, loss_dict

# class DomainContrastiveLoss(nn.Module):
#     def __init__(self, margin=0.5, alpha=1.0, beta=0.0, gamma=0.3, temperature=0.07):
#         """
#         改进的域对比损失
#         Args:
#             margin: 负约束的最小余弦距离
#             alpha: 正对齐损失权重
#             beta: 负分离损失权重
#             gamma: 类内紧凑性损失权重
#             temperature: 相似度分布锐化系数
#         """
#         super().__init__()
#         self.margin = margin
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.temperature = temperature

#     def forward(self, features, labels=None, cur_proto=None, prev_proto=None, class_weights=None):
#         B, D = features.size()
#         f_norm = F.normalize(features, p=2, dim=1)
        
#         loss_dict = {"loss_pos": 0.0, "loss_neg": 0.0, "loss_intra": 0.0}
#         total_loss = torch.tensor(0.0, device=features.device)

#         # ======================
#         # 1. 正对齐损失
#         # ======================
#         if cur_proto is not None and labels is not None:
#             # 检查原型有效性
#             proto_norms = torch.norm(cur_proto, p=2, dim=1)
#             valid_mask = (proto_norms > 1e-3) & (proto_norms < 100)  # 防止极端值
            
#             cur_proto_norm = F.normalize(cur_proto, p=2, dim=1)
#             pos_proto = cur_proto_norm[labels]
            
#             # 计算余弦相似度
#             pos_sim = (f_norm * pos_proto).sum(dim=1)
            
#             # 只对有效原型的样本计算损失
#             valid_samples = valid_mask[labels]
#             if valid_samples.sum() > 0:
#                 pos_loss = (1 - pos_sim).clamp(min=0)
#                 # 应用类别权重
#                 if class_weights is not None:
#                     sample_weights = class_weights[labels][valid_samples]
#                     loss_pos = (pos_loss * valid_samples * sample_weights).sum() / (sample_weights.sum() + 1e-8)
#                 else:
#                     loss_pos = (pos_loss * valid_samples).sum() / (valid_samples.sum() + 1e-8)
#                 total_loss += self.alpha * loss_pos
#                 loss_dict["loss_pos"] = loss_pos.item()
        
#         # ======================
#         # 2. 负分离损失
#         # ======================
#         if prev_proto is not None and prev_proto.size(0) > 0:
#             # 检查历史原型有效性
#             prev_norms = torch.norm(prev_proto, p=2, dim=1)
#             prev_valid = prev_norms > 1e-3
#             prev_proto_norm = F.normalize(prev_proto, p=2, dim=1)[prev_valid]
            
#             if prev_proto_norm.size(0) > 0:
#                 # 计算带温度的相似度矩阵
#                 sim_matrix = F.linear(f_norm, prev_proto_norm) / self.temperature
#                 cos_dist = 1.0 - sim_matrix
                
#                 # 类别感知的负样本选择
#                 if class_weights is not None:
#                     # 计算任务权重（简单示例：最近的任务权重更高）
#                     sample_weights = class_weights[labels]
#                     task_weights = torch.linspace(1.0, 0.5, steps=prev_proto_norm.size(0), device=features.device)
#                     # 组合权重
#                     combined_weights = sample_weights.unsqueeze(1) * task_weights.unsqueeze(0)
#                     # 应用权重到距离矩阵
#                     weighted_cos_dist = cos_dist * combined_weights
#                     # 关注最困难的负样本
#                     hard_neg_dist, _ = weighted_cos_dist.min(dim=1)
#                 else:
#                     hard_neg_dist, _ = cos_dist.min(dim=1)
#                 loss_neg = F.relu(self.margin - hard_neg_dist).mean()
                
#                 total_loss += self.beta * loss_neg
#                 loss_dict["loss_neg"] = loss_neg.item()
        
#         # ======================
#         # 3. 类内紧凑性损失
#         # ======================
#         if labels is not None and B > 1:
#             # 创建同类样本掩码
#             same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
#             same_class.fill_diagonal_(False)  # 排除自身
            
#             if same_class.sum() > 0:
#                 # 计算同类样本间的相似度
#                 intra_sim = torch.mm(f_norm, f_norm.t())
#                 intra_loss = (1 - intra_sim[same_class]).mean()
                
#                 total_loss += self.gamma * intra_loss
#                 loss_dict["loss_intra"] = intra_loss.item()
        
#         return total_loss, loss_dict

# class DomainContrastiveLoss(nn.Module):
#     def __init__(self, margin=0.5, alpha=1.0, beta=0.5, gamma=0.3, temperature=0.07, 
#                  adaptive_temp=True, smooth_loss=True, curriculum_learning=True):
#         """
#         改进的域对比损失 - 解决收敛问题
#         Args:
#             margin: 负约束的最小余弦距离
#             alpha: 正对齐损失权重
#             beta: 负分离损失权重
#             gamma: 类内紧凑性损失权重
#             temperature: 相似度分布锐化系数
#             adaptive_temp: 是否使用自适应温度
#             smooth_loss: 是否使用平滑损失
#             curriculum_learning: 是否使用课程学习
#         """
#         super().__init__()
#         self.margin = margin
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.temperature = temperature
#         self.adaptive_temp = adaptive_temp
#         self.smooth_loss = smooth_loss
#         self.curriculum_learning = curriculum_learning
        
#         # 自适应参数
#         self.register_buffer('step_count', torch.tensor(0))
#         self.register_buffer('temp_min', torch.tensor(0.01))
#         self.register_buffer('temp_max', torch.tensor(0.5))

#     def get_adaptive_temperature(self, similarities):
#         """自适应温度调整"""
#         if not self.adaptive_temp:
#             return self.temperature
        
#         # 根据相似度分布调整温度
#         sim_std = similarities.std()
#         # 相似度分布越集中，温度越高（软化分布）
#         adaptive_temp = torch.clamp(
#             self.temperature * (1 + sim_std), 
#             self.temp_min.to(sim_std.device), 
#             self.temp_max.to(sim_std.device)
#         )
#         return adaptive_temp

#     def curriculum_weight(self, step):
#         """课程学习权重调整"""
#         if not self.curriculum_learning:
#             return 1.0
        
#         # 训练初期降低负分离损失的权重
#         warmup_steps = 1000
#         if step < warmup_steps:
#             return 0.1 + 0.9 * (step / warmup_steps)
#         return 1.0

#     def infonce_loss(self, query, pos_key, neg_keys, temperature):
#         """InfoNCE损失 - 类似CLIP"""
#         # 计算正样本相似度
#         pos_sim = torch.sum(query * pos_key, dim=-1, keepdim=True) / temperature
        
#         # 计算负样本相似度
#         neg_sim = torch.mm(query, neg_keys.t()) / temperature
        
#         # 组合正负样本
#         logits = torch.cat([pos_sim, neg_sim], dim=1)
        
#         # 标签：正样本在第0位
#         labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
        
#         return F.cross_entropy(logits, labels)

#     def smooth_triplet_loss(self, anchor, positive, negatives, margin, temperature):
#         """平滑三元组损失"""
#         # 计算距离
#         pos_dist = 1 - F.cosine_similarity(anchor, positive, dim=1)
        
#         # 对所有负样本计算距离并进行软最小值
#         neg_dists = []
#         for neg in negatives:
#             neg_dist = 1 - F.cosine_similarity(anchor, neg, dim=1)
#             neg_dists.append(neg_dist)
        
#         if neg_dists:
#             neg_dist_stack = torch.stack(neg_dists, dim=1)
#             # 使用log-sum-exp技巧计算软最小值
#             weights = F.softmax(-neg_dist_stack / temperature, dim=1)
#             soft_min_neg_dist = torch.sum(weights * neg_dist_stack, dim=1)
#         else:
#             soft_min_neg_dist = torch.zeros_like(pos_dist)
        
#         # 平滑三元组损失
#         loss = F.relu(pos_dist - soft_min_neg_dist + margin)
#         return loss.mean()

#     def forward(self, features, labels=None, cur_proto=None, prev_proto=None, 
#                 prev_proto_labels=None, class_weights=None):
#         B, D = features.size()
#         f_norm = F.normalize(features, p=2, dim=1)
        
#         # 更新步数
#         self.step_count += 1
        
#         loss_dict = {"loss_pos": 0.0, "loss_neg": 0.0, "loss_intra": 0.0}
#         total_loss = torch.tensor(0.0, device=features.device)

#         # ======================
#         # 1. 正对齐损失 (改进版)
#         # ======================
#         if cur_proto is not None and labels is not None:
#             proto_norms = torch.norm(cur_proto, p=2, dim=1)
#             valid_mask = (proto_norms > 1e-3) & (proto_norms < 100)
            
#             cur_proto_norm = F.normalize(cur_proto, p=2, dim=1)
#             pos_proto = cur_proto_norm[labels]
            
#             # 使用cosine embedding loss而不是简单的1-cosine
#             pos_sim = F.cosine_similarity(f_norm, pos_proto, dim=1)
#             target = torch.ones_like(pos_sim)  # 目标相似度为1
            
#             valid_samples = valid_mask[labels]
#             if valid_samples.sum() > 0:
#                 # 使用cosine embedding loss
#                 pos_loss = F.cosine_embedding_loss(
#                     f_norm[valid_samples], 
#                     pos_proto[valid_samples], 
#                     target[valid_samples],
#                     reduction='mean'
#                 )
                
#                 if class_weights is not None:
#                     sample_weights = class_weights[labels][valid_samples].mean()
#                     pos_loss = pos_loss * sample_weights
                
#                 total_loss += self.alpha * pos_loss
#                 loss_dict["loss_pos"] = pos_loss.item()

#         # ======================
#         # 2. 负分离损失 (InfoNCE + 平滑版本)
#         # ======================
#         curr_beta = self.beta * self.curriculum_weight(self.step_count)
        
#         if curr_beta > 0:
#             # 方案1: InfoNCE风格的对比损失
#             if cur_proto is not None and labels is not None and self.smooth_loss:
#                 proto_norms = torch.norm(cur_proto, p=2, dim=1)
#                 valid_mask = (proto_norms > 1e-3) & (proto_norms < 100)
                
#                 if valid_mask.sum() > 1:  # 至少需要2个有效原型
#                     cur_proto_norm = F.normalize(cur_proto, p=2, dim=1)
                    
#                     # 为每个样本构建正负样本对
#                     infonce_losses = []
#                     for i, sample_label in enumerate(labels):
#                         if valid_mask[sample_label]:
#                             # 正样本：当前类别原型
#                             pos_proto = cur_proto_norm[sample_label:sample_label+1]
                            
#                             # 负样本：其他有效类别原型
#                             neg_mask = valid_mask.clone()
#                             neg_mask[sample_label] = False
                            
#                             if neg_mask.sum() > 0:
#                                 neg_protos = cur_proto_norm[neg_mask]
                                
#                                 # 自适应温度
#                                 all_sims = torch.cat([
#                                     F.cosine_similarity(f_norm[i:i+1], pos_proto, dim=1),
#                                     F.cosine_similarity(f_norm[i:i+1], neg_protos, dim=1)
#                                 ])
#                                 temp = self.get_adaptive_temperature(all_sims)
                                
#                                 # InfoNCE损失
#                                 infonce_loss = self.infonce_loss(
#                                     f_norm[i:i+1], pos_proto, neg_protos, temp
#                                 )
#                                 infonce_losses.append(infonce_loss)
                    
#                     if infonce_losses:
#                         loss_neg = torch.stack(infonce_losses).mean()
#                         total_loss += curr_beta * loss_neg
#                         loss_dict["loss_neg"] = loss_neg.item()
            
#             # 方案2: 平滑三元组损失（传统方式的改进）
#             elif not self.smooth_loss and cur_proto is not None and labels is not None:
#                 proto_norms = torch.norm(cur_proto, p=2, dim=1)
#                 valid_mask = (proto_norms > 1e-3) & (proto_norms < 100)
                
#                 if valid_mask.sum() > 1:
#                     cur_proto_norm = F.normalize(cur_proto, p=2, dim=1)
                    
#                     triplet_losses = []
#                     for i, sample_label in enumerate(labels):
#                         if valid_mask[sample_label]:
#                             # anchor: 当前样本
#                             anchor = f_norm[i:i+1]
#                             # positive: 当前类别原型
#                             positive = cur_proto_norm[sample_label:sample_label+1]
                            
#                             # negatives: 其他类别原型
#                             neg_mask = valid_mask.clone()
#                             neg_mask[sample_label] = False
                            
#                             if neg_mask.sum() > 0:
#                                 negatives = cur_proto_norm[neg_mask]
                                
#                                 # 自适应温度
#                                 all_sims = torch.cat([
#                                     F.cosine_similarity(anchor, positive, dim=1),
#                                     F.cosine_similarity(anchor, negatives, dim=1)
#                                 ])
#                                 temp = self.get_adaptive_temperature(all_sims)
                                
#                                 # 平滑三元组损失
#                                 triplet_loss = self.smooth_triplet_loss(
#                                     anchor, positive, [negatives[j:j+1] for j in range(negatives.size(0))],
#                                     self.margin, temp
#                                 )
#                                 triplet_losses.append(triplet_loss)
                    
#                     if triplet_losses:
#                         loss_neg = torch.stack(triplet_losses).mean()
#                         total_loss += curr_beta * loss_neg
#                         loss_dict["loss_neg"] = loss_neg.item()

#         # ======================
#         # 3. 类内紧凑性损失 (保持原有实现)
#         # ======================
#         if labels is not None and B > 1:
#             same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
#             same_class.fill_diagonal_(False)
            
#             if same_class.sum() > 0:
#                 intra_sim = torch.mm(f_norm, f_norm.t())
#                 intra_loss = (1 - intra_sim[same_class]).mean()
                
#                 total_loss += self.gamma * intra_loss
#                 loss_dict["loss_intra"] = intra_loss.item()

#         return total_loss, loss_dict

class DomainContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, alpha=1.0, beta=0.5, gamma=0.3, temperature=0.07, 
                 adaptive_temp=True, smooth_loss=True, curriculum_learning=True):
        """
        改进的域对比损失 - 解决收敛问题
        Args:
            margin: 负约束的最小余弦距离
            alpha: 正对齐损失权重
            beta: 负分离损失权重
            gamma: 类内紧凑性损失权重
            temperature: 相似度分布锐化系数
            adaptive_temp: 是否使用自适应温度
            smooth_loss: 是否使用平滑损失
            curriculum_learning: 是否使用课程学习
        """
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.adaptive_temp = adaptive_temp
        self.smooth_loss = smooth_loss
        self.curriculum_learning = curriculum_learning
        
        # 自适应参数
        self.register_buffer('step_count', torch.tensor(0))
        self.register_buffer('temp_min', torch.tensor(0.01))
        self.register_buffer('temp_max', torch.tensor(0.5))

    def get_adaptive_temperature(self, similarities):
        """自适应温度调整"""
        if not self.adaptive_temp:
            return self.temperature
        
        # 根据相似度分布调整温度
        sim_std = similarities.std()
        # 相似度分布越集中，温度越高（软化分布）
        adaptive_temp = torch.clamp(
            self.temperature * (1 + sim_std), 
            self.temp_min.to(sim_std.device), 
            self.temp_max.to(sim_std.device)
        )
        return adaptive_temp

    def curriculum_weight(self, step):
        """课程学习权重调整"""
        if not self.curriculum_learning:
            return 1.0
        
        # 训练初期降低负分离损失的权重
        warmup_steps = 1000
        if step < warmup_steps:
            return 0.1 + 0.9 * (step / warmup_steps)
        return 1.0

    def infonce_loss(self, query, pos_key, neg_keys, temperature):
        """InfoNCE损失 - 类似CLIP"""
        # 计算正样本相似度
        pos_sim = torch.sum(query * pos_key, dim=-1, keepdim=True) / temperature
        
        # 计算负样本相似度
        neg_sim = torch.mm(query, neg_keys.t()) / temperature
        
        # 组合正负样本
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        
        # 标签：正样本在第0位
        labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
        
        return F.cross_entropy(logits, labels)

    def smooth_triplet_loss(self, anchor, positive, negatives, margin, temperature):
        """平滑三元组损失"""
        # 计算距离
        pos_dist = 1 - F.cosine_similarity(anchor, positive, dim=1)
        
        # 对所有负样本计算距离并进行软最小值
        neg_dists = []
        for neg in negatives:
            neg_dist = 1 - F.cosine_similarity(anchor, neg, dim=1)
            neg_dists.append(neg_dist)
        
        if neg_dists:
            neg_dist_stack = torch.stack(neg_dists, dim=1)
            # 使用log-sum-exp技巧计算软最小值
            weights = F.softmax(-neg_dist_stack / temperature, dim=1)
            soft_min_neg_dist = torch.sum(weights * neg_dist_stack, dim=1)
        else:
            soft_min_neg_dist = torch.zeros_like(pos_dist)
        
        # 平滑三元组损失
        loss = F.relu(pos_dist - soft_min_neg_dist + margin)
        return loss.mean()

    def forward(self, features, labels=None, cur_proto=None, prev_proto=None, 
                prev_proto_labels=None, class_weights=None):
        B, D = features.size()
        f_norm = F.normalize(features, p=2, dim=1)
        
        # 更新步数
        self.step_count += 1
        
        loss_dict = {"loss_pos": 0.0, "loss_neg": 0.0, "loss_intra": 0.0}
        total_loss = torch.tensor(0.0, device=features.device)

        # ======================
        # 1. 正对齐损失 (改进版)
        # ======================
        if cur_proto is not None and labels is not None:
            proto_norms = torch.norm(cur_proto, p=2, dim=1)
            valid_mask = (proto_norms > 1e-3) & (proto_norms < 100)
            
            cur_proto_norm = F.normalize(cur_proto, p=2, dim=1)
            pos_proto = cur_proto_norm[labels]
            
            # 使用cosine embedding loss而不是简单的1-cosine
            pos_sim = F.cosine_similarity(f_norm, pos_proto, dim=1)
            target = torch.ones_like(pos_sim)  # 目标相似度为1
            
            valid_samples = valid_mask[labels]
            if valid_samples.sum() > 0:
                # 使用cosine embedding loss
                pos_loss = F.cosine_embedding_loss(
                    f_norm[valid_samples], 
                    pos_proto[valid_samples], 
                    target[valid_samples],
                    reduction='mean'
                )
                
                if class_weights is not None:
                    sample_weights = class_weights[labels][valid_samples].mean()
                    pos_loss = pos_loss * sample_weights
                
                total_loss += self.alpha * pos_loss
                loss_dict["loss_pos"] = pos_loss.item()

        # ======================
        # 2. 负分离损失 (InfoNCE + 平滑版本 + 跨域分离)
        # ======================
        curr_beta = self.beta * self.curriculum_weight(self.step_count)
        
        if curr_beta > 0:
            loss_neg_total = torch.tensor(0.0, device=features.device)
            neg_loss_count = 0
            
            # 2.1 当前域内的负分离损失
            if cur_proto is not None and labels is not None:
                proto_norms = torch.norm(cur_proto, p=2, dim=1)
                valid_mask = (proto_norms > 1e-3) & (proto_norms < 100)
                
                if valid_mask.sum() > 1:  # 至少需要2个有效原型
                    cur_proto_norm = F.normalize(cur_proto, p=2, dim=1)
                    
                    if self.smooth_loss:
                        # InfoNCE风格的域内对比损失
                        infonce_losses = []
                        for i, sample_label in enumerate(labels):
                            if valid_mask[sample_label]:
                                # 正样本：当前类别原型
                                pos_proto = cur_proto_norm[sample_label:sample_label+1]
                                
                                # 负样本：其他有效类别原型
                                neg_mask = valid_mask.clone()
                                neg_mask[sample_label] = False
                                
                                if neg_mask.sum() > 0:
                                    neg_protos = cur_proto_norm[neg_mask]
                                    
                                    # 自适应温度
                                    all_sims = torch.cat([
                                        F.cosine_similarity(f_norm[i:i+1], pos_proto, dim=1),
                                        F.cosine_similarity(f_norm[i:i+1], neg_protos, dim=1)
                                    ])
                                    temp = self.get_adaptive_temperature(all_sims)
                                    
                                    # InfoNCE损失
                                    infonce_loss = self.infonce_loss(
                                        f_norm[i:i+1], pos_proto, neg_protos, temp
                                    )
                                    infonce_losses.append(infonce_loss)
                        
                        if infonce_losses:
                            loss_neg_intra = torch.stack(infonce_losses).mean()
                            loss_neg_total += loss_neg_intra
                            neg_loss_count += 1
                    
                    else:
                        # 平滑三元组损失（域内）
                        triplet_losses = []
                        for i, sample_label in enumerate(labels):
                            if valid_mask[sample_label]:
                                anchor = f_norm[i:i+1]
                                positive = cur_proto_norm[sample_label:sample_label+1]
                                
                                neg_mask = valid_mask.clone()
                                neg_mask[sample_label] = False
                                
                                if neg_mask.sum() > 0:
                                    negatives = cur_proto_norm[neg_mask]
                                    
                                    all_sims = torch.cat([
                                        F.cosine_similarity(anchor, positive, dim=1),
                                        F.cosine_similarity(anchor, negatives, dim=1)
                                    ])
                                    temp = self.get_adaptive_temperature(all_sims)
                                    
                                    triplet_loss = self.smooth_triplet_loss(
                                        anchor, positive, [negatives[j:j+1] for j in range(negatives.size(0))],
                                        self.margin, temp
                                    )
                                    triplet_losses.append(triplet_loss)
                        
                        if triplet_losses:
                            loss_neg_intra = torch.stack(triplet_losses).mean()
                            loss_neg_total += loss_neg_intra
                            neg_loss_count += 1
            
            # 2.2 跨域的负分离损失（历史原型）
            if prev_proto is not None and prev_proto.size(0) > 0 and labels is not None:
                prev_norms = torch.norm(prev_proto, p=2, dim=1)
                prev_valid_mask = prev_norms > 1e-3
                
                if prev_valid_mask.sum() > 0:
                    prev_proto_norm = F.normalize(prev_proto, p=2, dim=1)[prev_valid_mask]
                    
                    if prev_proto_labels is not None:
                        # 已知历史原型标签 - 类别感知的跨域分离
                        valid_prev_labels = prev_proto_labels[prev_valid_mask]
                        cross_domain_losses = []
                        
                        for i, sample_label in enumerate(labels):
                            # 找到历史原型中与当前样本不同类别的原型
                            diff_class_mask = valid_prev_labels != sample_label
                            diff_class_protos = prev_proto_norm[diff_class_mask]
                            
                            if diff_class_protos.size(0) > 0:
                                if self.smooth_loss:
                                    # 使用InfoNCE风格，但权重较小
                                    # 构造虚拟正样本（当前样本自身）
                                    query = f_norm[i:i+1]
                                    
                                    # 计算与不同类历史原型的相似度
                                    cross_sim = torch.mm(query, diff_class_protos.t())
                                    temp = self.get_adaptive_temperature(cross_sim.flatten()) * 2  # 更高温度
                                    
                                    # 简化的对比损失：推动远离所有不同类历史原型
                                    cross_logits = cross_sim / temp
                                    # 目标是让所有相似度都很小
                                    cross_loss = F.logsumexp(cross_logits, dim=1).mean()
                                    cross_domain_losses.append(cross_loss)
                                else:
                                    # 使用margin loss，权重较小
                                    cross_sim = F.cosine_similarity(f_norm[i:i+1], diff_class_protos, dim=1)
                                    cross_dist = 1.0 - cross_sim
                                    hard_cross_dist = cross_dist.min()
                                    cross_loss = F.relu(self.margin - hard_cross_dist)
                                    cross_domain_losses.append(cross_loss)
                        
                        if cross_domain_losses:
                            loss_neg_cross = torch.stack(cross_domain_losses).mean()
                            # 跨域分离使用较小权重（0.3）
                            loss_neg_total += 0.3 * loss_neg_cross
                            neg_loss_count += 1
                    
                    else:
                        # 未知历史原型标签 - 保守的轻微分离
                        if self.smooth_loss:
                            # 对所有历史原型进行轻微的InfoNCE风格分离
                            cross_sim_matrix = torch.mm(f_norm, prev_proto_norm.t())
                            temp = self.get_adaptive_temperature(cross_sim_matrix.flatten()) * 3  # 更高温度，更保守
                            
                            cross_logits = cross_sim_matrix / temp
                            # 非常轻微的推斥力
                            cross_loss = torch.logsumexp(cross_logits, dim=1).mean()
                            # 使用很小的权重（0.1）避免错误分离同类
                            loss_neg_total += 0.1 * cross_loss
                            neg_loss_count += 1
                        else:
                            # 传统margin方式，非常保守
                            cross_sim_matrix = torch.mm(f_norm, prev_proto_norm.t())
                            cross_cos_dist = 1.0 - cross_sim_matrix
                            hard_cross_dist, _ = cross_cos_dist.min(dim=1)
                            cross_loss = F.relu(self.margin - hard_cross_dist).mean()
                            # 使用很小的权重（0.1）
                            loss_neg_total += 0.1 * cross_loss
                            neg_loss_count += 1
            
            # 计算最终的负分离损失
            if neg_loss_count > 0:
                loss_neg = loss_neg_total / neg_loss_count
                
                # 应用类别权重
                if class_weights is not None:
                    sample_weights = class_weights[labels].mean()
                    loss_neg = loss_neg * sample_weights
                
                total_loss += curr_beta * loss_neg
                loss_dict["loss_neg"] = loss_neg.item()

        # ======================
        # 3. 类内紧凑性损失 (保持原有实现)
        # ======================
        if labels is not None and B > 1:
            same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
            same_class.fill_diagonal_(False)
            
            if same_class.sum() > 0:
                intra_sim = torch.mm(f_norm, f_norm.t())
                intra_loss = (1 - intra_sim[same_class]).mean()
                
                total_loss += self.gamma * intra_loss
                loss_dict["loss_intra"] = intra_loss.item()

        return total_loss, loss_dict

# class DomainContrastiveLoss(nn.Module):
#     """
#     精简且修正后的域对比损失实现（可直接替换原实现）

#     设计要点：
#     - 使用矢量化 InfoNCE 风格实现（避免逐样本循环）作为域内负分离核心
#     - 正对齐使用逐样本 (1 - cosine) 并支持逐样本类别权重
#     - 历史原型（prev_proto）支持已知标签或未知标签两种情况；已知标签下会掩掉同类原型
#     - 自适应温度、课程学习、平滑项保留为可选，但实现更稳健
#     - 去除了低效的逐负样本循环，优化数值稳定性与权重应用方式

#     使用说明：
#     - cur_proto: Tensor[C, D]，第 i 行为类别 i 的当前原型（如不满足需外部提供映射）
#     - prev_proto: Tensor[P, D]，历史原型集合
#     - prev_proto_labels: Tensor[P] 可选，表示 prev_proto 的类别索引
#     - labels: Tensor[B]，每个样本对应的类别索引（0..C-1）

#     返回： total_loss (scalar), loss_dict
#     """

#     def __init__(self, margin=0.5, alpha=1.0, beta=0.5, gamma=0.3,
#                  temperature=0.07, adaptive_temp=True, smooth_loss=True,
#                  curriculum_learning=True, cross_domain_weight=0.1):
#         super().__init__()
#         self.margin = float(margin)
#         self.alpha = float(alpha)
#         self.beta = float(beta)
#         self.gamma = float(gamma)
#         self.temperature = float(temperature)
#         self.adaptive_temp = bool(adaptive_temp)
#         self.smooth_loss = bool(smooth_loss)
#         self.curriculum_learning = bool(curriculum_learning)
#         self.cross_domain_weight = float(cross_domain_weight)

#         # 使用普通 python int 跟踪步数，避免 buffer inplace 更新问题
#         self.step_count = 0
#         self.temp_min = 0.01
#         self.temp_max = 5.0

#     def get_adaptive_temperature(self, sims):
#         """根据相似度分布返回标量温度，sims 可以是 1D 或 2D tensor"""
#         if not self.adaptive_temp:
#             return torch.tensor(self.temperature, device=sims.device, dtype=sims.dtype)

#         sim_flat = sims.flatten()
#         # 若 sims 全相等会导致 std=0 -> temperature = base
#         sim_std = sim_flat.std(unbiased=False) if sim_flat.numel() > 1 else torch.tensor(0.0, device=sims.device)
#         adaptive = self.temperature * (1.0 + sim_std)
#         adaptive = torch.clamp(adaptive, self.temp_min, self.temp_max)
#         return adaptive

#     def curriculum_weight(self, step):
#         if not self.curriculum_learning:
#             return 1.0
#         warmup_steps = 1000
#         if step < warmup_steps:
#             return float(0.1 + 0.9 * (step / warmup_steps))
#         return 1.0

#     def forward(self, features, labels=None, cur_proto=None, prev_proto=None,
#                 prev_proto_labels=None, class_weights=None):
#         """
#         features: (B, D)
#         labels: (B,)  or None
#         cur_proto: (C, D) or None
#         prev_proto: (P, D) or None
#         prev_proto_labels: (P,) or None
#         class_weights: Tensor[C] or None (per-class scalar weight)
#         """
#         B, D = features.shape
#         device = features.device
#         f_norm = F.normalize(features, p=2, dim=1)
#         self.step_count += 1

#         total_loss = torch.tensor(0.0, device=device)
#         loss_dict = {"loss_pos": 0.0, "loss_neg": 0.0, "loss_intra": 0.0}

#         # ------------------
#         # 1) 正对齐 (per-sample 1 - cos) with per-sample weights
#         # ------------------
#         if cur_proto is not None and labels is not None:
#             cur_proto_norm = F.normalize(cur_proto, p=2, dim=1)  # (C, D)
#             C = cur_proto_norm.size(0)
#             # 检查有效原型
#             proto_norms = torch.norm(cur_proto, p=2, dim=1)
#             valid_mask = (proto_norms > 1e-6) & (proto_norms < 1e6)

#             # 逐样本的正原型（若某些标签对应的原型无效，将其视为不可计算）
#             valid_samples = valid_mask[labels]
#             if valid_samples.sum() > 0:
#                 pos_proto = cur_proto_norm[labels]  # (B, D)
#                 cos_sim = (f_norm * pos_proto).sum(dim=1)  # (B,)
#                 pos_loss_vec = (1.0 - cos_sim).clamp(min=0.0)  # (B,)

#                 if class_weights is not None:
#                     # class_weights: size C
#                     sample_weights = class_weights[labels].to(device)
#                 else:
#                     sample_weights = torch.ones(B, device=device)

#                 # 仅对 valid_samples 计算加权平均
#                 denom = (sample_weights * valid_samples.float()).sum() + 1e-12
#                 pos_loss = (pos_loss_vec * sample_weights * valid_samples.float()).sum() / denom

#                 total_loss = total_loss + self.alpha * pos_loss
#                 loss_dict["loss_pos"] = float(pos_loss.detach())

#         # ------------------
#         # 2) 负分离（域内 InfoNCE 风格） - 矢量化实现
#         #    使用 cur_proto（类别感知）作为候选正/负原型
#         # ------------------
#         curr_beta = self.beta * self.curriculum_weight(self.step_count)
#         loss_neg_terms = []

#         if curr_beta > 0 and cur_proto is not None and labels is not None:
#             # 准备 proto 与 mask
#             proto_norm = F.normalize(cur_proto, p=2, dim=1)  # (C, D)
#             proto_norms = torch.norm(cur_proto, p=2, dim=1)
#             valid_proto_mask = (proto_norms > 1e-6) & (proto_norms < 1e6)

#             # 若可用原型少于2则跳过内部 InfoNCE
#             if valid_proto_mask.sum() > 1:
#                 # logits: (B, C)
#                 # 注意数值稳定性：对无效 proto 列赋非常小的值
#                 logits = torch.matmul(f_norm, proto_norm.t())  # cosine similarities (B, C)
#                 # adaptive temperature based on logits distribution
#                 temp = float(self.get_adaptive_temperature(logits))
#                 logits = logits / temp

#                 # mask invalid prototypes
#                 invalid_cols = ~valid_proto_mask
#                 if invalid_cols.any():
#                     logits[:, invalid_cols] = -1e9

#                 # Now use cross-entropy where correct class index is labels
#                 # This naturally pulls up the positive proto (labels) and pushes down others
#                 # But to avoid accidentally treating identical-class old prototypes as negatives later,
#                 # we handle prev_proto separately
#                 ce_loss_per_sample = F.cross_entropy(logits, labels.to(device), reduction='none')

#                 # apply per-sample class weights
#                 if class_weights is not None:
#                     sample_w = class_weights[labels].to(device)
#                 else:
#                     sample_w = torch.ones(B, device=device)

#                 loss_neg_intra = (ce_loss_per_sample * sample_w).sum() / (sample_w.sum() + 1e-12)
#                 loss_neg_terms.append(loss_neg_intra)

#         # ------------------
#         # 2b) 跨域负分离（使用 prev_proto），已知标签时阻止同类原型作为负样本
#         # ------------------
#         if curr_beta > 0 and prev_proto is not None and prev_proto.size(0) > 0:
#             prev_norms = torch.norm(prev_proto, p=2, dim=1)
#             prev_valid_mask = (prev_norms > 1e-6) & (prev_norms < 1e6)

#             if prev_valid_mask.sum() > 0:
#                 prev_proto_norm = F.normalize(prev_proto, p=2, dim=1)
#                 prev_proto_norm = prev_proto_norm[prev_valid_mask]

#                 if prev_proto_labels is not None:
#                     prev_labels_valid = prev_proto_labels[prev_valid_mask]
#                     # logits (B, P_valid)
#                     logits_prev = torch.matmul(f_norm, prev_proto_norm.t())
#                     temp_prev = float(self.get_adaptive_temperature(logits_prev)) * 2.0
#                     logits_prev = logits_prev / temp_prev

#                     # 对于与当前样本同类的历史原型，我们将其 mask 掉（防止被当作负样本）
#                     # 构建 mask (B, P_valid)
#                     # prev_labels_valid: (P_valid,)
#                     mask_same = (labels.unsqueeze(1).to(device) == prev_labels_valid.unsqueeze(0).to(device))
#                     logits_prev = logits_prev.masked_fill(mask_same, -1e9)

#                     # 推远历史不同类原型的简单但稳定的做法：对每个样本计算 logsumexp 并平均
#                     # 这会鼓励这些相似度整体较低
#                     # 使用较小权重避免误伤
#                     lse_per_sample = torch.logsumexp(logits_prev, dim=1)  # (B,)

#                     if class_weights is not None:
#                         sample_w = class_weights[labels].to(device)
#                     else:
#                         sample_w = torch.ones(B, device=device)

#                     loss_cross = (lse_per_sample * sample_w).sum() / (sample_w.sum() + 1e-12)
#                     loss_neg_terms.append(self.cross_domain_weight * loss_cross)

#                 else:
#                     # 未知历史标签 - 更保守的做法：对所有历史原型使用 logsumexp 且权重更小
#                     logits_prev = torch.matmul(f_norm, prev_proto_norm.t())
#                     temp_prev = float(self.get_adaptive_temperature(logits_prev)) * 3.0
#                     logits_prev = logits_prev / temp_prev
#                     lse_per_sample = torch.logsumexp(logits_prev, dim=1)

#                     if class_weights is not None:
#                         sample_w = class_weights[labels].to(device)
#                     else:
#                         sample_w = torch.ones(B, device=device)

#                     loss_cross = (lse_per_sample * sample_w).sum() / (sample_w.sum() + 1e-12)
#                     # 更小权重
#                     loss_neg_terms.append(0.1 * loss_cross)

#         # combine negative terms
#         if len(loss_neg_terms) > 0:
#             loss_neg = torch.stack(loss_neg_terms).mean()
#             total_loss = total_loss + curr_beta * loss_neg
#             loss_dict["loss_neg"] = float(loss_neg.detach())

#         # ------------------
#         # 3) 类内紧凑性（保持原实现）
#         # ------------------
#         if labels is not None and B > 1:
#             same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
#             same_class.fill_diagonal_(False)
#             if same_class.sum() > 0:
#                 intra_sim = torch.matmul(f_norm, f_norm.t())
#                 intra_loss = (1.0 - intra_sim[same_class]).mean()
#                 total_loss = total_loss + self.gamma * intra_loss
#                 loss_dict["loss_intra"] = float(intra_loss.detach())

#         return total_loss, loss_dict


# class DomainContrastiveLoss(nn.Module):
#     def __init__(self, margin=0.5, alpha=1.0, beta=1.0):
#         """
#         域增量学习的正负对比损失 (自适应版本)
#         Args:
#             margin: 负约束的最小余弦距离 (0-1)
#             alpha: 正对齐损失权重
#             beta: 负分离损失权重
#         """
#         super().__init__()
#         self.margin = margin
#         self.alpha = alpha
#         self.beta = beta

#     def forward(self, features, labels=None, cur_proto=None, prev_proto=None):
#         """
#         Args:
#             features: 当前域样本特征 [B, D]
#             labels: 当前域样本类别标签 [B]，可选
#             cur_proto: 当前域原型 [n_cur_classes, D]，可选
#             prev_proto: 历史原型 [n_prev_classes, D]，可选
#         Returns:
#             total_loss: 标量 loss
#             dict_losses: 各部分 loss
#         """
#         B, D = features.size()
#         f_norm = F.normalize(features, p=2, dim=1)

#         # ======================
#         # 1. 正对齐 (可选，自适应)
#         # ======================
#         loss_pos = torch.tensor(0.0, device=features.device)
#         if cur_proto is not None and labels is not None:
#             cur_proto_norm = F.normalize(cur_proto, p=2, dim=1)

#             pos_proto = cur_proto_norm[labels]  # [B, D]

#             # 找出哪些 proto 是全 0（即无效原型）
#             valid_mask = (cur_proto[labels].abs().sum(dim=1) > 1e-8).float()

#             # 计算余弦相似度
#             pos_sim = (f_norm * pos_proto).sum(dim=1)  # [B]

#             # 只对有效原型的样本计算 loss
#             if valid_mask.sum() > 0:
#                 loss_pos = ((1.0 - pos_sim) * valid_mask).sum() / (valid_mask.sum() + 1e-8)

#         # ======================
#         # 2. 负分离 (可选)
#         # ======================
#         loss_neg = torch.tensor(0.0, device=features.device)
#         if prev_proto is not None and prev_proto.size(0) > 0:
#             prev_proto_norm = F.normalize(prev_proto, p=2, dim=1)
#             # sim_matrix = torch.matmul(f_norm, prev_proto_norm.T)  # [B, N_prev]
#             sim_matrix = F.linear(f_norm, prev_proto_norm)
#             cos_dist = 1.0 - sim_matrix
#             min_dist, _ = cos_dist.min(dim=1)
#             loss_neg = F.relu(self.margin - min_dist).mean()

#         # ======================
#         # 3. 总损失
#         # ======================
#         total_loss = self.alpha * loss_pos + self.beta * loss_neg

#         return total_loss, {"loss_pos": loss_pos.item(), "loss_neg": loss_neg.item()}

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


# def accuracy(y_pred, y_true, nb_old, init_cls=10, increment=10):
#     assert len(y_pred) == len(y_true), "Data length error."
#     all_acc = {}
#     all_acc["total"] = np.around(
#         (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
#     )

#     # Grouped accuracy, for initial classes
#     idxes = np.where(
#         np.logical_and(y_true >= 0, y_true < init_cls)
#     )[0]
#     label = "{}-{}".format(
#         str(0).rjust(2, "0"), str(init_cls - 1).rjust(2, "0")
#     )
#     all_acc[label] = np.around(
#         (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
#     )
#     # for incremental classes
#     if increment > 0:
#         for class_id in range(init_cls, np.max(y_true), increment):
#             idxes = np.where(
#                 np.logical_and(y_true >= class_id, y_true < class_id + increment)
#             )[0]
#             label = "{}-{}".format(
#                 str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
#             )
#             all_acc[label] = np.around(
#                 (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
#             )

#         # Old accuracy
#         idxes = np.where(y_true < nb_old)[0]

#         all_acc["old"] = (
#             0
#             if len(idxes) == 0
#             else np.around(
#                 (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
#             )
#         )

#     # New accuracy
#     idxes = np.where(y_true >= nb_old)[0]
#     all_acc["new"] = np.around(
#         (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
#     )

#     return all_acc

# def accuracy(y_pred, y_true, nb_old, init_cls=10, increment=10, current_task=0):
#     assert len(y_pred) == len(y_true), "Data length error."
#     all_acc = {}
#     all_acc["total"] = np.around(
#         (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
#     )
#     if isinstance(increment, list):
#         # 处理边界条件
#         increment = increment[:current_task+1] if (current_task+1 < len(increment)) else increment
#         # 生成动态任务边界
#         task_boundaries = [0]
#         for inc in increment:
#             task_boundaries.append(task_boundaries[-1] + inc)
            
#         # 动态分组评估
#         for i in range(len(task_boundaries)-1):
#             start = task_boundaries[i]
#             end = task_boundaries[i+1] - 1
#             idxes = np.where((y_true >= start) & (y_true <= end))[0]
#             label = f'task{i+1}_{start}-{end}'
#             acc_value = ((y_pred[idxes] % len(idxes)) == (y_true[idxes] % len(idxes))).sum() * 100
#             if len(idxes) == 0:
#                 all_acc[label] = 0.0
#                 continue
#             # 关键修复：使用任务实际增量计算偏移
#             acc_value = ((y_pred[idxes] - start) == (y_true[idxes] - start)).sum() * 100 / len(idxes)
#             all_acc[label] = np.around(acc_value, decimals=2)
            
#         # 新旧类别评估修复
#         idxes = np.where(y_true < nb_old)[0]
#         all_acc['old'] = 0 if len(idxes)==0 else np.around(
#             (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

#         idxes = np.where(y_true >= nb_old)[0]
#         all_acc['new'] = 0 if len(idxes)==0 else np.around(
#             ((y_pred[idxes] - nb_old) == (y_true[idxes] - nb_old)).sum() * 100 / len(idxes), decimals=2)
#     else:
#         # Grouped accuracy, for initial classes
#         idxes = np.where(
#             np.logical_and(y_true >= 0, y_true < init_cls)
#         )[0]
#         label = "{}-{}".format(
#             str(0).rjust(2, "0"), str(init_cls - 1).rjust(2, "0")
#         )
#         all_acc[label] = np.around(
#             (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
#         )
#         # for incremental classes
#         for class_id in range(init_cls, np.max(y_true), increment):
#             idxes = np.where(
#                 np.logical_and(y_true >= class_id, y_true < class_id + increment)
#             )[0]
#             label = "{}-{}".format(
#                 str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
#             )
#             all_acc[label] = np.around(
#                 (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
#             )

#         # Old accuracy
#         idxes = np.where(y_true < nb_old)[0]

#         all_acc["old"] = (
#             0
#             if len(idxes) == 0
#             else np.around(
#                 (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
#             )
#         )

#         # New accuracy
#         idxes = np.where(y_true >= nb_old)[0]
#         all_acc["new"] = np.around(
#             (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
#         )

#     return all_acc

# def accuracy(y_pred, y_true, nb_old, init_cls=10, increment=10, current_task=0):
#     assert len(y_pred) == len(y_true), "Data length error."
#     all_acc = {}

#     # 总体准确率
#     all_acc["total"] = np.around((y_pred == y_true).sum() * 100 / len(y_true), 2)

#     if isinstance(increment, list):
#         # 处理动态任务边界
#         increment = increment[:current_task+1] if (current_task+1 < len(increment)) else increment
#         task_boundaries = [0]
#         for inc in increment:
#             task_boundaries.append(task_boundaries[-1] + inc)

#         # 每个任务的分组准确率
#         for i in range(len(task_boundaries)-1):
#             start, end = task_boundaries[i], task_boundaries[i+1]
#             idxes = np.where((y_true >= start) & (y_true < end))[0]
#             if len(idxes) == 0:
#                 all_acc[f'task{i+1}_{start}-{end-1}'] = 0.0
#                 continue
#             acc_value = (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes)
#             all_acc[f'task{i+1}_{start}-{end-1}'] = np.around(acc_value, 2)

#         # 新旧类别准确率
#         idxes = np.where(y_true < nb_old)[0]
#         all_acc['old'] = 0 if len(idxes) == 0 else np.around(
#             (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), 2
#         )

#         idxes = np.where(y_true >= nb_old)[0]
#         all_acc['new'] = 0 if len(idxes) == 0 else np.around(
#             (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), 2
#         )

#     else:
#         # 初始类评估
#         idxes = np.where((y_true >= 0) & (y_true < init_cls))[0]
#         label = f"{0:02d}-{init_cls-1:02d}"
#         all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), 2)

#         # 增量类分组
#         for class_id in range(init_cls, np.max(y_true)+1, increment):
#             idxes = np.where((y_true >= class_id) & (y_true < class_id + increment))[0]
#             label = f"{class_id:02d}-{class_id+increment-1:02d}"
#             all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), 2)

#         # 新旧类
#         idxes = np.where(y_true < nb_old)[0]
#         all_acc["old"] = 0 if len(idxes)==0 else np.around(
#             (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), 2
#         )
#         idxes = np.where(y_true >= nb_old)[0]
#         all_acc["new"] = 0 if len(idxes)==0 else np.around(
#             (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), 2
#         )

#     return all_acc

def accuracy(y_pred, y_true, nb_old, init_cls=10, increment=10, current_task=0, normalize_cm=False):
    """
    综合分类评估：返回准确率、任务级别acc、新旧类表现、误分类分析、混淆矩阵等
    Args:
        y_pred: 预测标签 (np.array 或 torch.Tensor)
        y_true: 真实标签 (np.array 或 torch.Tensor)
        nb_old: 旧类数目（用于区分 old/new）
        init_cls: 初始类别数
        increment: 每次增量类别数，可以是 int 或 list
        current_task: 当前任务编号
        normalize_cm: 是否归一化混淆矩阵
    Returns:
        metrics: dict 包含多种评估指标
    """
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    assert len(y_pred) == len(y_true), "Data length error."
    metrics = {}

    # -------------------------
    # 1. 总体准确率
    metrics["total"] = np.around((y_pred == y_true).sum() * 100 / len(y_true), 2)

    # -------------------------
    # 2. 分任务准确率
    if isinstance(increment, list):
        increment = increment[:current_task+1] if (current_task+1 < len(increment)) else increment
        task_boundaries = [0]
        for inc in increment:
            task_boundaries.append(task_boundaries[-1] + inc)

        for i in range(len(task_boundaries)-1):
            start, end = task_boundaries[i], task_boundaries[i+1] - 1
            idxes = np.where((y_true >= start) & (y_true <= end))[0]
            label = f'task{i+1}_{start}-{end}'
            if len(idxes) == 0:
                metrics[label] = 0.0
                continue
            acc_value = ((y_pred[idxes] - start) == (y_true[idxes] - start)).sum() * 100 / len(idxes)
            metrics[label] = np.around(acc_value, 2)
    else:
        # 初始类 + 增量类分组
        idxes = np.where((y_true >= 0) & (y_true < init_cls))[0]
        metrics[f"0-{init_cls-1}"] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), 2)
        for class_id in range(init_cls, np.max(y_true)+1, increment):
            idxes = np.where((y_true >= class_id) & (y_true < class_id + increment))[0]
            label = f"{class_id}-{class_id+increment-1}"
            if len(idxes) == 0: 
                metrics[label] = 0.0
                continue
            metrics[label] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), 2)

    # -------------------------
    # 3. old/new 准确率
    idx_old, idx_new = np.where(y_true < nb_old)[0], np.where(y_true >= nb_old)[0]
    metrics["old"] = 0 if len(idx_old)==0 else np.around((y_pred[idx_old] == y_true[idx_old]).sum() * 100 / len(idx_old), 2)
    metrics["new"] = 0 if len(idx_new)==0 else np.around((y_pred[idx_new] == y_true[idx_new]).sum() * 100 / len(idx_new), 2)

    # -------------------------
    # 4. old→new / new→old 误分类率
    if len(idx_old) > 0:
        wrong_old_to_new = np.sum((y_pred[idx_old] >= nb_old) & (y_pred[idx_old] != y_true[idx_old]))
        metrics['old_to_new_err'] = np.around(wrong_old_to_new / len(idx_old) * 100, 2)
    if len(idx_new) > 0:
        wrong_new_to_old = np.sum((y_pred[idx_new] < nb_old) & (y_pred[idx_new] != y_true[idx_new]))
        metrics['new_to_old_err'] = np.around(wrong_new_to_old / len(idx_new) * 100, 2)

    # -------------------------
    # 5. 每类 Precision / Recall / F1
    report = classification_report(y_true, y_pred, digits=2, output_dict=True)
    metrics["per_class"] = report

    # -------------------------
    # 6. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    if normalize_cm:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    metrics["confusion_matrix"] = cm

    return metrics


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)


def write_domain_img_file2txt(root_path, domain_name: str, extensions=['jpg', 'png', 'jpeg']):
    """
    Write all image paths and labels to a txt file,
    :param root_path: specific data path, e.g. /home/xxx/data/office-home
    :param domain_name: e.g. 'Art'
    """
    if os.path.exists(os.path.join(root_path, domain_name + '_all.txt')):
        return

    img_paths = []
    domain_path = os.path.join(root_path, domain_name)

    cl_dirs = os.listdir(domain_path)

    for cl_idx in range(len(cl_dirs)):

        cl_name = cl_dirs[cl_idx]
        cl_path = os.path.join(domain_path, cl_name)

        for img_file in os.listdir(cl_path):
            if img_file.split('.')[-1] in extensions:
                img_paths.append(os.path.join(domain_name, cl_name, img_file) + ' ' + str(cl_idx) + '\n')

    with open(os.path.join(root_path, domain_name + '_all.txt'), 'w') as f:
        for img_path in img_paths:
            f.write(img_path)

    # return img_paths
    
    

def split_domain_txt2txt(root_path, domain_name: str, train_ratio=0.7, seed=1993):
    """
    Split a txt file to train and test txt files.
    :param root_path: specific data path, e.g. /home/xxx/data/office-home
    :param domain_name: e.g. 'Art'
    :param train_ratio: ratio of train data
    """
    if os.path.exists(os.path.join(root_path, domain_name + '_train.txt')):
        return

    print("Split {} data to train and test txt files.".format(domain_name))
    np.random.seed(seed)
    print("Set numpy random seed to {}.".format(seed))

    with open(os.path.join(root_path, domain_name + '_all.txt'), 'r') as f:
        lines = f.readlines()
        np.random.shuffle(lines)
        train_lines = lines[:int(len(lines) * train_ratio)]
        test_lines = lines[int(len(lines) * train_ratio):]

    with open(os.path.join(root_path, domain_name + '_train.txt'), 'w') as f:
        for line in train_lines:
            f.write(line)

    with open(os.path.join(root_path, domain_name + '_test.txt'), 'w') as f:
        for line in test_lines:
            f.write(line)
            
            