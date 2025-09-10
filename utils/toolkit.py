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

class DomainContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, alpha=1.0, beta=0.0, gamma=0.3, temperature=0.07):
        """
        改进的域对比损失
        Args:
            margin: 负约束的最小余弦距离
            alpha: 正对齐损失权重
            beta: 负分离损失权重
            gamma: 类内紧凑性损失权重
            temperature: 相似度分布锐化系数
        """
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature

    def forward(self, features, labels=None, cur_proto=None, prev_proto=None, class_weights=None):
        B, D = features.size()
        f_norm = F.normalize(features, p=2, dim=1)
        
        loss_dict = {"loss_pos": 0.0, "loss_neg": 0.0, "loss_intra": 0.0}
        total_loss = torch.tensor(0.0, device=features.device)

        # ======================
        # 1. 正对齐损失
        # ======================
        if cur_proto is not None and labels is not None:
            # 检查原型有效性
            proto_norms = torch.norm(cur_proto, p=2, dim=1)
            valid_mask = (proto_norms > 1e-3) & (proto_norms < 100)  # 防止极端值
            
            cur_proto_norm = F.normalize(cur_proto, p=2, dim=1)
            pos_proto = cur_proto_norm[labels]
            
            # 计算余弦相似度
            pos_sim = (f_norm * pos_proto).sum(dim=1)
            
            # 只对有效原型的样本计算损失
            valid_samples = valid_mask[labels]
            if valid_samples.sum() > 0:
                pos_loss = (1 - pos_sim).clamp(min=0)
                # 应用类别权重
                if class_weights is not None:
                    sample_weights = class_weights[labels][valid_samples]
                    loss_pos = (pos_loss * valid_samples * sample_weights).sum() / (sample_weights.sum() + 1e-8)
                else:
                    loss_pos = (pos_loss * valid_samples).sum() / (valid_samples.sum() + 1e-8)
                total_loss += self.alpha * loss_pos
                loss_dict["loss_pos"] = loss_pos.item()
        
        # ======================
        # 2. 负分离损失
        # ======================
        if prev_proto is not None and prev_proto.size(0) > 0:
            # 检查历史原型有效性
            prev_norms = torch.norm(prev_proto, p=2, dim=1)
            prev_valid = prev_norms > 1e-3
            prev_proto_norm = F.normalize(prev_proto, p=2, dim=1)[prev_valid]
            
            if prev_proto_norm.size(0) > 0:
                # 计算带温度的相似度矩阵
                sim_matrix = F.linear(f_norm, prev_proto_norm) / self.temperature
                cos_dist = 1.0 - sim_matrix
                
                # 类别感知的负样本选择
                if class_weights is not None:
                    # 计算任务权重（简单示例：最近的任务权重更高）
                    sample_weights = class_weights[labels]
                    task_weights = torch.linspace(1.0, 0.5, steps=prev_proto_norm.size(0), device=features.device)
                    # 组合权重
                    combined_weights = sample_weights.unsqueeze(1) * task_weights.unsqueeze(0)
                    # 应用权重到距离矩阵
                    weighted_cos_dist = cos_dist * combined_weights
                    # 关注最困难的负样本
                    hard_neg_dist, _ = weighted_cos_dist.min(dim=1)
                else:
                    hard_neg_dist, _ = cos_dist.min(dim=1)
                loss_neg = F.relu(self.margin - hard_neg_dist).mean()
                
                total_loss += self.beta * loss_neg
                loss_dict["loss_neg"] = loss_neg.item()
        
        # ======================
        # 3. 类内紧凑性损失
        # ======================
        if labels is not None and B > 1:
            # 创建同类样本掩码
            same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
            same_class.fill_diagonal_(False)  # 排除自身
            
            if same_class.sum() > 0:
                # 计算同类样本间的相似度
                intra_sim = torch.mm(f_norm, f_norm.t())
                intra_loss = (1 - intra_sim[same_class]).mean()
                
                total_loss += self.gamma * intra_loss
                loss_dict["loss_intra"] = intra_loss.item()
        
        return total_loss, loss_dict

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
            
            