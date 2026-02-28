import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Optional, Union

class DomainContrastiveLoss(nn.Module):
    """
    精简且修正后的域对比损失实现（可直接替换原实现）

    设计要点：
    - 使用矢量化 InfoNCE 风格实现（避免逐样本循环）作为域内负分离核心
    - 正对齐使用逐样本 (1 - cosine) 并支持逐样本类别权重
    - 历史原型（prev_proto）支持已知标签或未知标签两种情况；已知标签下会掩掉同类原型
    - 自适应温度、课程学习、平滑项保留为可选，但实现更稳健
    - 去除了低效的逐负样本循环，优化数值稳定性与权重应用方式

    使用说明：
    - cur_proto: Tensor[C, D]，第 i 行为类别 i 的当前原型（如不满足需外部提供映射）
    - prev_proto: Tensor[P, D]，历史原型集合
    - prev_proto_labels: Tensor[P] 可选，表示 prev_proto 的类别索引
    - labels: Tensor[B]，每个样本对应的类别索引（0..C-1）

    返回： total_loss (scalar), loss_dict
    """

    def __init__(self, margin=0.5, alpha=1.0, beta=0.5, gamma=0.3,
                 temperature=0.07, adaptive_temp=True, smooth_loss=True,
                 curriculum_learning=True, cross_domain_weight=0.0):
        super().__init__()
        self.margin = float(margin)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.temperature = float(temperature)
        self.adaptive_temp = bool(adaptive_temp)
        self.smooth_loss = bool(smooth_loss)
        self.curriculum_learning = bool(curriculum_learning)
        self.cross_domain_weight = float(cross_domain_weight)

        # 使用普通 python int 跟踪步数，避免 buffer inplace 更新问题
        self.step_count = 0
        self.temp_min = 0.01
        self.temp_max = 5.0

    def get_adaptive_temperature(self, sims):
        """根据相似度分布返回标量温度，sims 可以是 1D 或 2D tensor"""
        if not self.adaptive_temp:
            return torch.tensor(self.temperature, device=sims.device, dtype=sims.dtype)

        sim_flat = sims.flatten()
        # 若 sims 全相等会导致 std=0 -> temperature = base
        sim_std = sim_flat.std(unbiased=False) if sim_flat.numel() > 1 else torch.tensor(0.0, device=sims.device)
        adaptive = self.temperature * (1.0 + sim_std)
        adaptive = torch.clamp(adaptive, self.temp_min, self.temp_max)
        return adaptive

    def curriculum_weight(self, step):
        if not self.curriculum_learning:
            return 1.0
        warmup_steps = 1000
        if step < warmup_steps:
            return float(0.1 + 0.9 * (step / warmup_steps))
        return 1.0

    def forward(self, features, labels=None, cur_proto=None, prev_proto=None,
                prev_proto_labels=None, class_weights=None):
        """
        features: (B, D)
        labels: (B,)  or None
        cur_proto: (C, D) or None
        prev_proto: (P, D) or None
        prev_proto_labels: (P,) or None
        class_weights: Tensor[C] or None (per-class scalar weight)
        """
        B, D = features.shape
        device = features.device
        f_norm = F.normalize(features, p=2, dim=1)
        self.step_count += 1

        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {"loss_pos": 0.0, "loss_neg": 0.0, "loss_intra": 0.0}

        # ------------------
        # 1) 正对齐 (per-sample 1 - cos) with per-sample weights
        # ------------------
        if cur_proto is not None and labels is not None:
            cur_proto_norm = F.normalize(cur_proto, p=2, dim=1)  # (C, D)
            C = cur_proto_norm.size(0)
            # 检查有效原型
            proto_norms = torch.norm(cur_proto, p=2, dim=1)
            valid_mask = (proto_norms > 1e-6) & (proto_norms < 1e6)

            # 逐样本的正原型（若某些标签对应的原型无效，将其视为不可计算）
            valid_samples = valid_mask[labels]
            if valid_samples.sum() > 0:
                pos_proto = cur_proto_norm[labels]  # (B, D)
                cos_sim = (f_norm * pos_proto).sum(dim=1)  # (B,)
                pos_loss_vec = (1.0 - cos_sim).clamp(min=0.0)  # (B,)

                if class_weights is not None:
                    # class_weights: size C
                    sample_weights = class_weights[labels].to(device)
                else:
                    sample_weights = torch.ones(B, device=device)

                # 仅对 valid_samples 计算加权平均
                denom = (sample_weights * valid_samples.float()).sum() + 1e-12
                pos_loss = (pos_loss_vec * sample_weights * valid_samples.float()).sum() / denom

                total_loss = total_loss + self.alpha * pos_loss
                loss_dict["loss_pos"] = float(pos_loss.detach())

        # ------------------
        # 2) 负分离（域内 InfoNCE 风格） - 矢量化实现
        #    使用 cur_proto（类别感知）作为候选正/负原型
        # ------------------
        curr_beta = self.beta * self.curriculum_weight(self.step_count)
        loss_neg_terms = torch.tensor(0.0, device=device)

        if curr_beta > 0 and cur_proto is not None and labels is not None:
            # 准备 proto 与 mask
            proto_norm = F.normalize(cur_proto, p=2, dim=1)  # (C, D)
            proto_norms = torch.norm(cur_proto, p=2, dim=1)
            valid_proto_mask = (proto_norms > 1e-6) & (proto_norms < 1e6)

            # 若可用原型少于2则跳过内部 InfoNCE
            if valid_proto_mask.sum() > 1:
                # logits: (B, C)
                # 注意数值稳定性：对无效 proto 列赋非常小的值
                logits = torch.matmul(f_norm, proto_norm.t())  # cosine similarities (B, C)
                # adaptive temperature based on logits distribution
                temp = float(self.get_adaptive_temperature(logits))
                # temp = self.get_adaptive_temperature(logits)
                logits = logits / temp

                # mask invalid prototypes
                invalid_cols = ~valid_proto_mask
                if invalid_cols.any():
                    logits[:, invalid_cols] = -1e9

                # Now use cross-entropy where correct class index is labels
                # This naturally pulls up the positive proto (labels) and pushes down others
                # But to avoid accidentally treating identical-class old prototypes as negatives later,
                # we handle prev_proto separately
                ce_loss_per_sample = F.cross_entropy(logits, labels.to(device), reduction='none')

                # apply per-sample class weights
                if class_weights is not None:
                    sample_w = class_weights[labels].to(device)
                else:
                    sample_w = torch.ones(B, device=device)

                loss_neg_intra = (ce_loss_per_sample * sample_w).sum() / (sample_w.sum() + 1e-12)
                loss_neg_terms += loss_neg_intra

        # # ------------------
        # 2b) 跨域负分离（使用 prev_proto），已知标签时阻止同类原型作为负样本
        # ------------------
        if curr_beta > 0 and prev_proto is not None and prev_proto.size(0) > 0:
            prev_norms = torch.norm(prev_proto, p=2, dim=1)
            prev_valid_mask = (prev_norms > 1e-6) & (prev_norms < 1e6)

            if prev_valid_mask.sum() > 0:
                prev_proto_norm = F.normalize(prev_proto, p=2, dim=1)
                prev_proto_norm = prev_proto_norm[prev_valid_mask]

                if prev_proto_labels is not None:
                    prev_labels_valid = prev_proto_labels[prev_valid_mask]
                    # logits (B, P_valid)
                    logits_prev = torch.matmul(f_norm, prev_proto_norm.t())
                    temp_prev = float(self.get_adaptive_temperature(logits_prev)) * 2.0
                    logits_prev = logits_prev / temp_prev

                    # 对于与当前样本同类的历史原型，我们将其 mask 掉（防止被当作负样本）
                    # 构建 mask (B, P_valid)
                    # prev_labels_valid: (P_valid,)
                    mask_same = (labels.unsqueeze(1).to(device) == prev_labels_valid.unsqueeze(0).to(device))
                    logits_prev = logits_prev.masked_fill(mask_same, -1e9)

                    # 推远历史不同类原型的简单但稳定的做法：对每个样本计算 logsumexp 并平均
                    # 这会鼓励这些相似度整体较低
                    # 使用较小权重避免误伤
                    lse_per_sample = torch.logsumexp(logits_prev, dim=1)  # (B,)

                    if class_weights is not None:
                        sample_w = class_weights[labels].to(device)
                    else:
                        sample_w = torch.ones(B, device=device)

                    loss_cross = (lse_per_sample * sample_w).sum() / (sample_w.sum() + 1e-12)
                    loss_neg_terms += self.cross_domain_weight * loss_cross
                else:
                    # 未知历史标签 - 更保守的做法：对所有历史原型使用 logsumexp 且权重更小
                    logits_prev = torch.matmul(f_norm, prev_proto_norm.t())
                    temp_prev = float(self.get_adaptive_temperature(logits_prev)) * 3.0
                    logits_prev = logits_prev / temp_prev
                    lse_per_sample = torch.logsumexp(logits_prev, dim=1)

                    if class_weights is not None:
                        sample_w = class_weights[labels].to(device)
                    else:
                        sample_w = torch.ones(B, device=device)

                    loss_cross = (lse_per_sample * sample_w).sum() / (sample_w.sum() + 1e-12)
                    # 更小权重
                    loss_neg_terms += self.cross_domain_weight * loss_cross
                    
        # combine negative terms
        total_loss = total_loss + curr_beta * loss_neg_terms
        loss_dict["loss_neg"] = float(loss_neg_terms.detach())

        # ------------------
        # 3) 类内紧凑性（保持原实现）
        # ------------------
        if cur_proto is not None and B > 1: # and labels is not None
            same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
            same_class.fill_diagonal_(False)
            if same_class.sum() > 0:
                intra_sim = torch.matmul(f_norm, f_norm.t())
                intra_loss = (1.0 - intra_sim[same_class]).mean()
                total_loss = total_loss + self.gamma * intra_loss
                loss_dict["loss_intra"] = float(intra_loss.detach())

        return total_loss, loss_dict


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
            
def weighted_federated_averaging(cur_adapter: nn.ModuleList,
                               old_adapter_list: nn.ModuleList, 
                               target_params: Optional[Union[str, List[str]]] = "lora_A",
                               weights: Optional[List[float]] = None,
                               layer_indices: Optional[List[int]] = None,
                               adapter_indices: Optional[List[int]] = None):
    """
    加权版本的联邦参数平均，可以为不同的历史adapter分配不同权重
    
    Args:
        weights: 历史adapter的权重列表，长度应等于old_adapter_list的长度
                如果为None，则使用均等权重
        target_params: 要平均的参数名，None表示所有参数
    """
    
    if len(old_adapter_list) == 0:
        print("没有历史adapter，跳过参数平均")
        return
    
    # 设置权重
    if weights is None:
        weights = [1.0 / len(old_adapter_list)] * len(old_adapter_list)
    else:
        assert len(weights) == len(old_adapter_list), "权重数量必须等于历史adapter数量"
        # 归一化权重
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
    
    # 确保target_params是列表格式，或处理None的情况
    if target_params is None:
        # 获取第一个非Identity模块的所有参数名
        target_params = []
        for layer_idx in range(len(cur_adapter)):
            for adapter_idx in range(len(cur_adapter[layer_idx])):
                cur_module = cur_adapter[layer_idx][adapter_idx]
                if not isinstance(cur_module, nn.Identity):
                    param_names = [name for name, param in cur_module.named_parameters()]
                    target_params.extend(param_names)
                    break
            if target_params:
                break
        target_params = list(set(target_params))
        print(f"检测到的所有参数: {target_params}")
    elif isinstance(target_params, str):
        target_params = [target_params]
    
    # 默认处理所有层和adapter
    if layer_indices is None:
        layer_indices = list(range(len(cur_adapter)))
    if adapter_indices is None:
        adapter_indices = list(range(len(cur_adapter[0])))
    
    print(f"开始加权联邦平均，权重: {weights}")
    print(f"处理参数: {target_params}")
    
    # 遍历指定的层和adapter
    for layer_idx in layer_indices:
        for adapter_idx in adapter_indices:
            cur_module = cur_adapter[layer_idx][adapter_idx]
            
            # 跳过Identity模块
            if isinstance(cur_module, nn.Identity):
                continue
            
            # 遍历要平均的参数
            for param_name in target_params:
                if not hasattr(cur_module, param_name):
                    print(f"警告: 层{layer_idx}的adapter{adapter_idx}没有属性 {param_name}")
                    continue
                
                cur_param = getattr(cur_module, param_name)
                
                # 情况1: target_name是Parameter
                if isinstance(cur_param, nn.Parameter):
                    # 收集并加权平均历史参数
                    weighted_sum = torch.zeros_like(cur_param.data)
                    valid_count = 0
                    
                    for i, old_adapter in enumerate(old_adapter_list):
                        old_module = old_adapter[layer_idx][adapter_idx]
                        if (not isinstance(old_module, nn.Identity) and 
                            hasattr(old_module, param_name)):
                            old_param = getattr(old_module, param_name)
                            weighted_sum += weights[i] * old_param.data
                            valid_count += 1
                
                    if valid_count > 0:
                        # 更新当前参数为加权平均结果
                        with torch.no_grad():
                            cur_param.data = weighted_sum.clone()
                        
                        print(f"已更新 层{layer_idx}-adapter{adapter_idx}-{param_name} (加权平均)")
                # 情况2: target_name是子模块(如Linear层)
                elif isinstance(cur_param, nn.Module):
                    # 对子模块的所有参数进行平均
                    for param_name, cur_param in cur_param.named_parameters():
                        # 收集并加权平均历史参数
                        weighted_sum = torch.zeros_like(cur_param.data)
                        valid_count = 0
                        
                        for i, old_adapter in enumerate(old_adapter_list):
                            old_module = old_adapter[layer_idx][adapter_idx]
                            if (not isinstance(old_module, nn.Identity) and 
                                hasattr(old_module, param_name)):
                                old_param = getattr(old_module, param_name)
                                weighted_sum += weights[i] * old_param.data
                                valid_count += 1
                        
                        if valid_count > 0:
                            # 更新当前参数为加权平均结果
                            with torch.no_grad():
                                cur_param.data = weighted_sum.clone()
                            
                            print(f"已更新 层{layer_idx}-adapter{adapter_idx}-{param_name} (加权平均)")
                else:
                    print(f"警告: {cur_param} 既不是Parameter也不是Module，跳过")

def selective_federated_averaging(cur_adapter: nn.ModuleList,
                                old_adapter_list: nn.ModuleList,
                                target_params: Optional[Union[str, List[str]]] = "lora_A", 
                                similarity_threshold: float = 0.8,
                                layer_indices: Optional[List[int]] = None,
                                adapter_indices: Optional[List[int]] = None):
    """
    基于相似性的选择性联邦平均，只融合相似度高的历史adapter参数
    
    Args:
        similarity_threshold: 相似度阈值，只有相似度超过此值的历史参数才参与平均
        target_params: 要平均的参数名，None表示所有参数
    """
    
    if len(old_adapter_list) == 0:
        print("没有历史adapter，跳过参数平均")
        return
    
    if isinstance(target_params, str):
        target_params = [target_params]
    elif target_params is None:
        # 获取第一个非Identity模块的所有参数名
        target_params = []
        for layer_idx in range(len(cur_adapter)):
            for adapter_idx in range(len(cur_adapter[layer_idx])):
                cur_module = cur_adapter[layer_idx][adapter_idx]
                if not isinstance(cur_module, nn.Identity):
                    param_names = [name for name, param in cur_module.named_parameters()]
                    target_params.extend(param_names)
                    break
            if target_params:
                break
        target_params = list(set(target_params))
        print(f"检测到的所有参数: {target_params}")
    
    if layer_indices is None:
        layer_indices = list(range(len(cur_adapter)))
    if adapter_indices is None:
        adapter_indices = list(range(len(cur_adapter[0])))
    
    print(f"开始选择性联邦平均，相似度阈值: {similarity_threshold}")
    
    for layer_idx in layer_indices:
        for adapter_idx in adapter_indices:
            cur_module = cur_adapter[layer_idx][adapter_idx]
            
            if isinstance(cur_module, nn.Identity):
                continue
            
            for param_name in target_params:
                if not hasattr(cur_module, param_name):
                    print(f"警告: 层{layer_idx}的adapter{adapter_idx}没有属性 {param_name}")
                    continue
                
                cur_param = getattr(cur_module, param_name)

                if isinstance(cur_param, nn.Parameter):
                # 计算与历史参数的相似度并选择性平均
                    similar_params = []
                    for old_adapter in old_adapter_list:
                        old_module = old_adapter[layer_idx][adapter_idx]
                        if (not isinstance(old_module, nn.Identity) and 
                            hasattr(old_module, param_name)):
                            old_param = getattr(old_module, param_name)
                            
                            # 计算余弦相似度
                            similarity = torch.cosine_similarity(
                                cur_param.data.flatten(), 
                                old_param.data.flatten(), 
                                dim=0
                            ).item()
                            
                            if similarity >= similarity_threshold:
                                similar_params.append(old_param.data.clone())
                    
                    if similar_params:
                        # 平均相似的参数
                        avg_similar = torch.stack(similar_params, dim=0).mean(dim=0)
                        
                        with torch.no_grad():
                            # 50-50混合当前参数和相似参数的平均
                            cur_param.data = 0.5 * cur_param.data + 0.5 * avg_similar
                        
                        print(f"层{layer_idx}-adapter{adapter_idx}-{param_name}: "
                            f"使用了{len(similar_params)}个相似参数进行平均")
                # 情况2: target_name是子模块(如Linear层)
                elif isinstance(cur_param, nn.Module):
                    for param_name, cur_param in cur_param.named_parameters():
                        similar_params = []
                        for old_adapter in old_adapter_list:
                            old_module = old_adapter[layer_idx][adapter_idx]
                            if (not isinstance(old_module, nn.Identity) and 
                                hasattr(old_module, param_name)):
                                old_param = getattr(old_module, param_name)
                                
                                # 计算余弦相似度
                                similarity = torch.cosine_similarity(
                                    cur_param.data.flatten(), 
                                    old_param.data.flatten(), 
                                    dim=0
                                ).item()
                                
                                if similarity >= similarity_threshold:
                                    similar_params.append(old_param.data.clone())
                        
                        if similar_params:
                            # 平均相似的参数
                            avg_similar = torch.stack(similar_params, dim=0).mean(dim=0)
                            
                            with torch.no_grad():
                                # 50-50混合当前参数和相似参数的平均
                                cur_param.data = 0.5 * cur_param.data + 0.5 * avg_similar
                            
                            print(f"层{layer_idx}-adapter{adapter_idx}-{param_name}: "
                                f"使用了{len(similar_params)}个相似参数进行平均")
                else:
                    print(f"警告: {cur_param} 既不是Parameter也不是Module，跳过")
                        
                    
def federated_adapter_averaging(cur_adapter: nn.ModuleList, 
                              old_adapter_list: nn.ModuleList,
                              target_params: Optional[Union[str, List[str]]] = "lora_A",
                              alpha: float = 0.5,
                              layer_indices: Optional[List[int]] = None,
                              adapter_indices: Optional[List[int]] = None):
    """
    使用联邦学习参数平均的思想融合历史adapter知识到当前adapter
    
    Args:
        cur_adapter: 当前域的adapter (ModuleList)
        old_adapter_list: 历史域的adapter列表 (ModuleList of ModuleList)
        target_params: 要平均的参数名，可以是字符串或列表，None表示所有参数
                      例如: "lora_A", "lora_B", ["lora_A", "lora_B"], None
        alpha: 融合权重，cur_param = alpha * cur_param + (1-alpha) * avg_old_params
        layer_indices: 指定要处理的层索引，None表示所有层
        adapter_indices: 指定要处理的adapter索引，None表示所有adapter
    """
    
    if len(old_adapter_list) == 0:
        print("没有历史adapter，跳过参数平均")
        return
    
    # 确保target_params是列表格式，或处理None的情况
    if target_params is None:
        # 获取第一个非Identity模块的所有参数名
        target_params = []
        for layer_idx in range(len(cur_adapter)):
            for adapter_idx in range(len(cur_adapter[layer_idx])):
                cur_module = cur_adapter[layer_idx][adapter_idx]
                if not isinstance(cur_module, nn.Identity):
                    # 获取该模块的所有参数名
                    param_names = [name for name, param in cur_module.named_parameters()]
                    submodule_names = [name for name, module in cur_module.named_children()]
                    target_params.extend(param_names + submodule_names)
                    target_params.extend(param_names)
                    break
            if target_params:  # 找到参数后跳出
                break
        
        # 去重
        target_params = list(set(target_params))
        print(f"检测到的所有参数: {target_params}")
        
    elif isinstance(target_params, str):
        target_params = [target_params]
    
    # 默认处理所有层和adapter
    if layer_indices is None:
        layer_indices = list(range(len(cur_adapter)))
    if adapter_indices is None:
        adapter_indices = list(range(len(cur_adapter[0])))
    
    print(f"开始联邦平均，处理参数: {target_params}")
    print(f"处理层: {layer_indices}, 处理adapter: {adapter_indices}")
    print(f"历史adapter数量: {len(old_adapter_list)}, 融合权重alpha: {alpha}")
    
    # 遍历指定的层和adapter
    for layer_idx in layer_indices:
        for adapter_idx in adapter_indices:
            cur_module = cur_adapter[layer_idx][adapter_idx]
            
            # 跳过Identity模块
            if isinstance(cur_module, nn.Identity):
                continue
            
            # 遍历要平均的参数
            for target_name in target_params:
                if not hasattr(cur_module, target_name):
                    print(f"警告: 层{layer_idx}的adapter{adapter_idx}没有参数 {target_name}")
                    continue
                
                target_attr = getattr(cur_module, target_name)
                
                # 情况1: target_name是Parameter
                if isinstance(target_attr, nn.Parameter):
                    # 收集所有历史adapter的对应参数
                    old_params = []
                    for old_adapter in old_adapter_list:
                        old_module = old_adapter[layer_idx][adapter_idx]
                        if (not isinstance(old_module, nn.Identity) and 
                            hasattr(old_module, target_name)):
                            old_param = getattr(old_module, target_name)
                            if isinstance(old_param, nn.Parameter):
                                old_params.append(old_param.data.clone())
                    
                    if old_params:
                        # 计算历史参数的平均值
                        avg_old_param = torch.stack(old_params, dim=0).mean(dim=0)
                        
                        # 联邦平均: 当前参数 = α * 当前参数 + (1-α) * 平均历史参数
                        with torch.no_grad():
                            target_attr.data = alpha * target_attr.data + (1 - alpha) * avg_old_param
                        
                        print(f"已更新参数 层{layer_idx}-adapter{adapter_idx}-{target_name}")
                
                # 情况2: target_name是子模块(如Linear层)
                elif isinstance(target_attr, nn.Module):
                    # 对子模块的所有参数进行平均
                    for param_name, cur_param in target_attr.named_parameters():
                        # 收集历史对应子模块的同名参数
                        old_params = []
                        for old_adapter in old_adapter_list:
                            old_module = old_adapter[layer_idx][adapter_idx]
                            if (not isinstance(old_module, nn.Identity) and 
                                hasattr(old_module, target_name)):
                                old_submodule = getattr(old_module, target_name)
                                if isinstance(old_submodule, nn.Module):
                                    # 获取对应参数
                                    old_param_dict = dict(old_submodule.named_parameters())
                                    if param_name in old_param_dict:
                                        old_params.append(old_param_dict[param_name].data.clone())
                        
                        if old_params:
                            # 计算历史参数的平均值
                            avg_old_param = torch.stack(old_params, dim=0).mean(dim=0)
                            
                            # 联邦平均
                            with torch.no_grad():
                                cur_param.data = alpha * cur_param.data + (1 - alpha) * avg_old_param
                            
                            # print(f"已更新子模块参数 层{layer_idx}-adapter{adapter_idx}-{target_name}.{param_name}")
                
                else:
                    print(f"警告: {target_name} 既不是Parameter也不是Module，跳过")