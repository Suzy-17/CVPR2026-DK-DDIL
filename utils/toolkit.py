import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, alpha=1.0, beta=1.0):
        """
        域增量学习的正负对比损失 (自适应版本)
        Args:
            margin: 负约束的最小余弦距离 (0-1)
            alpha: 正对齐损失权重
            beta: 负分离损失权重
        """
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta

    def forward(self, features, labels=None, cur_proto=None, prev_proto=None):
        """
        Args:
            features: 当前域样本特征 [B, D]
            labels: 当前域样本类别标签 [B]，可选
            cur_proto: 当前域原型 [n_cur_classes, D]，可选
            prev_proto: 历史原型 [n_prev_classes, D]，可选
        Returns:
            total_loss: 标量 loss
            dict_losses: 各部分 loss
        """
        B, D = features.size()
        f_norm = F.normalize(features, p=2, dim=1)

        # ======================
        # 1. 正对齐 (可选，自适应)
        # ======================
        loss_pos = torch.tensor(0.0, device=features.device)
        if cur_proto is not None and labels is not None:
            cur_proto_norm = F.normalize(cur_proto, p=2, dim=1)

            pos_proto = cur_proto_norm[labels]  # [B, D]

            # 找出哪些 proto 是全 0（即无效原型）
            valid_mask = (cur_proto[labels].abs().sum(dim=1) > 1e-8).float()

            # 计算余弦相似度
            pos_sim = (f_norm * pos_proto).sum(dim=1)  # [B]

            # 只对有效原型的样本计算 loss
            if valid_mask.sum() > 0:
                loss_pos = ((1.0 - pos_sim) * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        # ======================
        # 2. 负分离 (可选)
        # ======================
        loss_neg = torch.tensor(0.0, device=features.device)
        if prev_proto is not None and prev_proto.size(0) > 0:
            prev_proto_norm = F.normalize(prev_proto, p=2, dim=1)
            sim_matrix = torch.matmul(f_norm, prev_proto_norm.T)  # [B, N_prev]
            cos_dist = 1.0 - sim_matrix
            min_dist, _ = cos_dist.min(dim=1)
            loss_neg = F.relu(self.margin - min_dist).mean()

        # ======================
        # 3. 总损失
        # ======================
        total_loss = self.alpha * loss_pos + self.beta * loss_neg

        return total_loss, {"loss_pos": loss_pos.item(), "loss_neg": loss_neg.item()}

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

def accuracy(y_pred, y_true, nb_old, init_cls=10, increment=10, current_task=0):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )
    if isinstance(increment, list):
        # 处理边界条件
        increment = increment[:current_task+1] if (current_task+1 < len(increment)) else increment
        # 生成动态任务边界
        task_boundaries = [0]
        for inc in increment:
            task_boundaries.append(task_boundaries[-1] + inc)
            
        # 动态分组评估
        for i in range(len(task_boundaries)-1):
            start = task_boundaries[i]
            end = task_boundaries[i+1] - 1
            idxes = np.where((y_true >= start) & (y_true <= end))[0]
            label = f'task{i+1}_{start}-{end}'
            acc_value = ((y_pred[idxes] % len(idxes)) == (y_true[idxes] % len(idxes))).sum() * 100
            if len(idxes) == 0:
                all_acc[label] = 0.0
                continue
            # 关键修复：使用任务实际增量计算偏移
            acc_value = ((y_pred[idxes] - start) == (y_true[idxes] - start)).sum() * 100 / len(idxes)
            all_acc[label] = np.around(acc_value, decimals=2)
            
        # 新旧类别评估修复
        idxes = np.where(y_true < nb_old)[0]
        all_acc['old'] = 0 if len(idxes)==0 else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

        idxes = np.where(y_true >= nb_old)[0]
        all_acc['new'] = 0 if len(idxes)==0 else np.around(
            ((y_pred[idxes] - nb_old) == (y_true[idxes] - nb_old)).sum() * 100 / len(idxes), decimals=2)
    else:
        # Grouped accuracy, for initial classes
        idxes = np.where(
            np.logical_and(y_true >= 0, y_true < init_cls)
        )[0]
        label = "{}-{}".format(
            str(0).rjust(2, "0"), str(init_cls - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
        # for incremental classes
        for class_id in range(init_cls, np.max(y_true), increment):
            idxes = np.where(
                np.logical_and(y_true >= class_id, y_true < class_id + increment)
            )[0]
            label = "{}-{}".format(
                str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
            )
            all_acc[label] = np.around(
                (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
            )

        # Old accuracy
        idxes = np.where(y_true < nb_old)[0]

        all_acc["old"] = (
            0
            if len(idxes) == 0
            else np.around(
                (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
            )
        )

        # New accuracy
        idxes = np.where(y_true >= nb_old)[0]
        all_acc["new"] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    return all_acc

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
            
            