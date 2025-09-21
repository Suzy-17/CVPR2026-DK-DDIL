import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
import timm
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
# from timm.models.registry import register_model
import torch.nn.functional as F
import numpy as np
# import logging
# import os
# from collections import OrderedDict
import torch
import copy
# import random



# class Adapter_lora(nn.Module):
#     def __init__(self,
#                  config=None,
#                  d_model=None,
#                  bottleneck=None,
#                  dropout=0.0,
#                  init_option="bert",
#                  adapter_scalar="1.0",
#                  adapter_layernorm_option="in"):
#         super().__init__()
#         self.random_orth = True

#         self.n_embd = config.d_model if d_model is None else d_model
#         self.down_size = config.attn_bn if bottleneck is None else bottleneck

#         self.lora_A = nn.Linear(self.down_size, self.n_embd, bias=False)
#         self.lora_B = nn.Linear(self.n_embd, self.down_size, bias=False)

#         if self.random_orth:
#             random_matrix = torch.rand(self.n_embd, self.down_size)
#             q, r = torch.linalg.qr(random_matrix)
#             with torch.no_grad():
#                 self.lora_B.weight.copy_(q.T)
#             scaling_factor = 1.  # You can adjust this value if needed
#             self.lora_B.weight.data *= scaling_factor
#         else:
#             with torch.no_grad():
#                 nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))

#         if init_option == "bert":
#             raise NotImplementedError
#         elif init_option == "lora":
#             with torch.no_grad():
#                 nn.init.zeros_(self.lora_A.weight)
#         else:
#             raise NotImplementedError

#     def forward(self, x):
#         inter_x = self.lora_B(x)
#         out = self.lora_A(inter_x)
#         return out

# class Adapter_lora(nn.Module):
#     def __init__(self, 
#                  config=None,
#                  d_model=None,
#                  bottleneck=None,
#                  dropout=0.0,
#                  init_option="bert",
#                  adapter_scalar="1.0",
#                  adapter_layernorm_option="in",
#                  max_rank: int = 64,
#                  min_rank: int = 4,
#                  init_scale: float = 0.02,
#                  temp: float = 0.5,
#                  reg_alpha: float = 0.1):
#         super().__init__()
#         self.n_embd = config.d_model if d_model is None else d_model
#         self.max_rank = max_rank
#         self.min_rank = min_rank
#         self.temp = temp
        
#         # 可学习参数
#         self.rank_scores = nn.Parameter(torch.ones(max_rank))
#         nn.init.normal_(self.rank_scores, mean=0.0, std=0.02)
#         self.register_buffer('active_mask', torch.ones(max_rank, dtype=torch.bool))
        
#         # 调整矩阵维度定义
#         self.lora_A = nn.Linear(self.n_embd, max_rank, bias=False)  # [n_embd, max_rank]
#         self.lora_B = nn.Linear(max_rank, self.n_embd, bias=False)  # [max_rank, n_embd]
        
#         # 正交初始化校验
#         assert init_scale > 0, "初始化尺度必须为正数"
#         nn.init.orthogonal_(self.lora_A.weight)
#         nn.init.normal_(self.lora_B.weight, std=init_scale)
        
#         # 动态正则化参数
#         self.reg_alpha = reg_alpha
#         self.current_rank = max_rank

#     def get_active_components(self):
#         # 带温度参数的Gumbel-Softmax
#         if self.training:
#             logits = self.rank_scores / self.temp
#             probs = F.gumbel_softmax(logits, tau=self.temp, hard=False)
#             mask = probs > 0.5
#         else:
#             mask = self.rank_scores > 0.5
        
#         active_rank = torch.sum(mask).clamp(
#             min=self.min_rank,
#             max=min(self.max_rank, mask.shape[0])
#         )
#         return mask, active_rank

#     def forward(self, x):
#         mask, current_rank = self.get_active_components()
#         self.current_rank = current_rank.item()
        
#         # 前向传播修正
#         A = self.lora_A.weight[mask, :]  # [active_rank, n_embd]
#         B = self.lora_B.weight[:, mask]  # [n_embd, active_rank]
        
#         # 维度校验断言
#         assert A.shape[0] == B.shape[1], \
#             f"激活秩不匹配 A:{A.shape[0]} vs B:{B.shape[1]}"
        
#         # 保持梯度流的矩阵运算
#         return (x @ B) @ A.T  # 保持维度一致性

#     def regularization_loss(self):
#         # 对数缩放的正则化
#         ratio = self.max_rank / max(1, self.current_rank)
#         reg_coef = self.reg_alpha * torch.log(torch.tensor(ratio, device=self.rank_scores.device))

#         return reg_coef * torch.sum(torch.sigmoid(self.rank_scores))

#     @torch.no_grad()
#     def prune_parameters(self):
#         # 保持计算图的剪枝
#         self.active_mask = self.rank_scores > 0.5
#         new_A = nn.Parameter(self.lora_A.weight[:, self.active_mask])
#         new_B = nn.Parameter(self.lora_B.weight[self.active_mask, :])
        
#         # 替换参数并重置
#         self.lora_A = nn.Linear(self.n_embd, int(torch.sum(self.active_mask)), bias=False)
#         self.lora_B = nn.Linear(int(torch.sum(self.active_mask)), self.n_embd, bias=False)
#         self.lora_A.weight = new_A
#         self.lora_B.weight = new_B
        
#         # 重置参数
#         self.max_rank = int(torch.sum(self.active_mask))
#         self.rank_scores = nn.Parameter(torch.ones(self.max_rank, device=self.rank_scores.device))
        
#     def clip_gradients(self, max_norm=1.0):
#         # 梯度裁剪
#         torch.nn.utils.clip_grad_norm_(self.rank_scores, max_norm)

# class Adapter_lora(nn.Module):
#     """
#     改进版 AdapterLoRA:
#       - 不对 A/B 做 normalize（保留线性表达能力）
#       - dynamic scaling 使用当前有效秩（current_effective_rank）
#       - 增量感知正则化：可以把当前状态 commit 为 base，之后只对超出 base 的激活部分惩罚
#       - STE (straight-through) 在训练时使用硬/软 mask
#       - safe prune -> 替换为更小的线性层
#     """
#     def __init__(self,  
#                  config=None, 
#                  d_model=None, 
#                  bottleneck=None, 
#                  dropout=0.0, 
#                  init_option="bert", 
#                  adapter_scalar="1.0", 
#                  adapter_layernorm_option="in", 
#                  max_rank: int = 64, 
#                  min_rank: int = 4, 
#                  init_scale: float = 0.02, 
#                  temp: float = 2, 
#                  reg_alpha: float = 0.1,
#                  lora_alpha: float = 16.0):
#         super().__init__()

#         if config is not None and d_model is None:
#             d_model = config.d_model
#         assert d_model is not None, "d_model must be provided via arg or config"

#         self.n_embd = d_model
#         self.max_rank = max_rank
#         self.min_rank = max(min_rank, 1)
#         self.temp = max(float(temp), 1e-6)
#         self.reg_alpha = float(reg_alpha)
#         self.lora_alpha = float(lora_alpha)
#         self.init_scale = float(init_scale)
#         self.init_option = init_option

#         # 可学习的 rank_scores (控制哪些 rank 被激活)
#         self.rank_scores = nn.Parameter(torch.zeros(self.max_rank))
#         # 初始化为小随机值
#         with torch.no_grad():
#             self.rank_scores.normal_(mean=0.0, std=0.01)

#         # 基线 active ratio（用于增量感知正则化）
#         # 初始化为 0（表示没有已占用的 capacity）
#         self.register_buffer("base_active_ratio", torch.tensor(0.0))

#         # LoRA 矩阵：A: n_embd -> max_rank, B: max_rank -> n_embd
#         # 使用 nn.Linear 方便管理权重形状和 device
#         self.lora_A = nn.Linear(self.n_embd, self.max_rank, bias=False)
#         self.lora_B = nn.Linear(self.max_rank, self.n_embd, bias=False)

#         # 初始化 A/B
#         # A: small random / orthonormal-ish 有时更稳定； B: zeros (LoRA paper style)
#         with torch.no_grad():
#             # A: random normal scaled
#             nn.init.normal_(self.lora_A.weight, mean=0.0, std=self.init_scale)
#             if self.init_option == "lora":
#                 nn.init.zeros_(self.lora_B.weight)
#             else:
#                 nn.init.normal_(self.lora_B.weight, mean=0.0, std=self.init_scale)

#         # Dropout for LoRA input
#         self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

#         # 用于推理时快速判断
#         self.register_buffer("active_mask", torch.ones(self.max_rank, dtype=torch.bool))

#         # 当前有效 rank（数值；训练期间会更新）
#         self.current_effective_rank = self.max_rank

#     def _get_masks_and_rank(self):
#         """
#         返回 (mask, active_indices_bool, effective_rank_tensor)
#         mask: float mask 用于训练时乘权 (STE: forward is hard, backward uses soft)
#         active_indices_bool: boolean tensor 指出哪些维度被硬激活（用于 prune 或 eval mode 下切片）
#         effective_rank_tensor: scalar tensor 当前 rank（用于 scaling）
#         """
#         device = self.rank_scores.device
#         if self.training:
#             # soft scores for differentiability
#             soft = torch.sigmoid(self.rank_scores / self.temp)
#             hard = (soft > 0.5).float()
#             # STE: forward uses hard (discrete), backward uses soft
#             mask = hard.detach() + (soft - soft.detach())
#             active_bool = (hard > 0.5)
#         else:
#             # eval/deterministic: 用 sigmoid threshold
#             soft = torch.sigmoid(self.rank_scores)
#             active_bool = (soft > 0.5)
#             mask = active_bool.float()

#         # 保证至少 min_rank 被激活
#         active_count = int(active_bool.sum().item())
#         if active_count < self.min_rank:
#             # 选择 top-k
#             topk_vals, topk_idx = torch.topk(self.rank_scores, self.min_rank)
#             # 构建 hard active_bool
#             new_active_bool = torch.zeros_like(active_bool, dtype=torch.bool, device=device)
#             new_active_bool[topk_idx] = True

#             if self.training:
#                 # 构建 STE-friendly mask: 对 topk 扩充软值
#                 soft = torch.sigmoid(self.rank_scores / self.temp)
#                 new_mask = torch.zeros_like(mask, device=device)
#                 # for indices in topk, set mask as soft but ensure forward sees 1 for these
#                 # using STE trick: hard for forward, soft for backward
#                 hard_top = torch.zeros_like(mask, device=device)
#                 hard_top[topk_idx] = 1.0
#                 new_mask = hard_top.detach() + (soft - soft.detach())
#             else:
#                 new_mask = new_active_bool.float()

#             mask = new_mask
#             active_bool = new_active_bool
#             active_count = int(self.min_rank)

#         effective_rank = torch.tensor(active_count, dtype=torch.float32, device=device)
#         return mask, active_bool, effective_rank

#     def forward(self, x: torch.Tensor):
#         """
#         x: [batch, seq_len, n_embd]
#         returns: LoRA output with same shape [batch, seq_len, n_embd]
#         """
#         if x.ndim != 3:
#             raise ValueError(f"AdapterLoRA expects x with ndim==3 (B, L, C), got {x.ndim}")

#         mask, active_bool, effective_rank = self._get_masks_and_rank()
#         # update internal record
#         self.current_effective_rank = int(effective_rank.item())

#         # dynamic scaling: use current effective rank (>=1)
#         denom = float(max(1.0, self.current_effective_rank))
#         scaling = (self.lora_alpha / denom)

#         if self.current_effective_rank <= 0:
#             return torch.zeros_like(x)

#         # Weighted A/B
#         # mask shape: (max_rank,), A.weight shape: (max_rank, n_embd)
#         A_weighted = self.lora_A.weight * mask.unsqueeze(1)  # (max_rank, n_embd)
#         B_weighted = self.lora_B.weight * mask.unsqueeze(0)  # (n_embd, max_rank)

#         # apply dropout to input
#         x_dropout = self.lora_dropout(x)  # (B, L, C)

#         # compute: hidden = x @ A_weighted.T -> (B, L, max_rank)
#         hidden = F.linear(x_dropout, A_weighted)  # weight shape (out=max_rank, in=n_embd)
#         # output = hidden @ B_weighted.T -> (B, L, n_embd)
#         output = F.linear(hidden, B_weighted)  # B_weighted shape (out=n_embd, in=max_rank)

#         return output * scaling

#     def regularization_loss(self):
#         """
#         增量感知正则化：
#           - active_ratio = sum(sigmoid(rank_scores)) / max_rank
#           - penalize only the超出 base_active_ratio 的部分
#           - reg = reg_alpha * max(0, active_ratio - base_active_ratio)
#         """
#         with torch.no_grad():
#             sigmoid_scores = torch.sigmoid(self.rank_scores)
#             active_ratio = float(sigmoid_scores.sum().item() / max(1, self.max_rank))

#         # penalize growth beyond base
#         excess = max(0.0, active_ratio - float(self.base_active_ratio.item()))
#         return torch.tensor(self.reg_alpha * excess, device=self.rank_scores.device, dtype=torch.float32)

#     def commit_current_as_base(self):
#         """
#         把当前的 sigmoid(rank_scores) 平均作为 base_active_ratio（用于后续域增量训练）
#         意味着之后只会对超出这部分的活跃度进行正则化（鼓励复用已有容量）
#         """
#         with torch.no_grad():
#             sigmoid_scores = torch.sigmoid(self.rank_scores)
#             active_ratio = sigmoid_scores.sum().item() / max(1, self.max_rank)
#             self.base_active_ratio.fill_(active_ratio)

#     @torch.no_grad()
#     def prune_parameters(self):
#         """
#         根据当前 sigmoid(rank_scores) > 0.5 来剪枝并替换为更小的线性层。
#         保证至少 self.min_rank 被保留。
#         """
#         device = self.rank_scores.device
#         sigmoid_scores = torch.sigmoid(self.rank_scores)
#         active_mask = sigmoid_scores > 0.5
#         active_count = int(active_mask.sum().item())
#         if active_count < self.min_rank:
#             _, top_idx = torch.topk(sigmoid_scores, self.min_rank)
#             new_mask = torch.zeros_like(active_mask, dtype=torch.bool, device=device)
#             new_mask[top_idx] = True
#             active_mask = new_mask
#             active_count = self.min_rank

#         if active_count == 0:
#             # nothing to prune
#             return

#         # create new smaller layers
#         new_A = nn.Linear(self.n_embd, active_count, bias=False).to(device)
#         new_B = nn.Linear(active_count, self.n_embd, bias=False).to(device)

#         # copy weights from active indices
#         # lora_A.weight shape: (max_rank, n_embd)
#         new_A.weight.data.copy_(self.lora_A.weight[active_mask, :].clone())
#         # lora_B.weight shape: (n_embd, max_rank) -> we want columns corresponding to active_mask
#         new_B.weight.data.copy_(self.lora_B.weight[:, active_mask].clone())

#         # replace
#         self.lora_A = new_A
#         self.lora_B = new_B

#         # update bookkeeping
#         self.max_rank = active_count
#         # reset rank_scores to ones -> learn which of the remaining dims to keep
#         self.rank_scores = nn.Parameter(torch.zeros(self.max_rank, device=device))
#         with torch.no_grad():
#             self.rank_scores.normal_(mean=0.0, std=0.01)
#         # active_mask buffer
#         self.register_buffer("active_mask", torch.ones(self.max_rank, dtype=torch.bool, device=device))
#         # clamp base_active_ratio to new scale (normalize)
#         with torch.no_grad():
#             self.base_active_ratio.fill_(min(float(self.base_active_ratio.item()), 1.0))

#     def get_effective_rank(self):
#         with torch.no_grad():
#             sigmoid_scores = torch.sigmoid(self.rank_scores)
#             return int((sigmoid_scores > 0.5).sum().item())

#     def clip_rank_gradients(self, max_norm=1.0):
#         """方便在训练循环中仅裁剪 rank_scores 的梯度"""
#         if self.rank_scores.grad is not None:
#             torch.nn.utils.clip_grad_norm_([self.rank_scores], max_norm)

# from typing import Optional
# class Adapter_lora(nn.Module):
#     def __init__(self, 
#                  config=None,
#                  d_model: Optional[int] = None,
#                  bottleneck: Optional[int] = None,
#                  dropout: float = 0.0,
#                  init_option: str = "lora",
#                  adapter_scalar: float = 1.0,
#                  adapter_layernorm_option="in",
#                  max_rank: int = 64,
#                  min_rank: int = 8,
#                  init_scale: float = 0.02,
#                  temp: float = 2.0,
#                  reg_alpha: float = 0.1,
#                  lora_alpha: float = 16.0,
#                  ):
#         """
#         改进版LoRA适配器，支持动态秩调整和域增量学习
        
#         参数:
#         - d_model: 输入特征维度
#         - max_rank: 最大秩
#         - min_rank: 最小秩 (必须 >= 1)
#         - lora_alpha: LoRA缩放因子
#         - domain_id: 当前适配器所属领域ID
#         """
#         super().__init__()
#         self.n_embd = config.d_model if d_model is None else d_model
#         self.max_rank = max_rank
#         self.min_rank = max(1, min_rank)  # 确保最小秩至少为1
#         self.temp = max(float(temp), 1e-2)
#         self.lora_alpha = lora_alpha
#         # self.domain_id = domain_id
#         self.reg_alpha = reg_alpha
        
#         # 秩选择参数 - 使用Sigmoid门控
#         self.rank_scores = nn.Parameter(torch.ones(max_rank))
#         nn.init.normal_(self.rank_scores, mean=0.0, std=0.01)
        
#         # LoRA矩阵 - 使用更高效的初始化
#         self.lora_A = nn.Linear(self.n_embd, max_rank, bias=False)
#         self.lora_B = nn.Linear(max_rank, self.n_embd, bias=False)
        
#         # 初始化方案
#         if init_option == "bert":
#             # 类BERT初始化
#             nn.init.normal_(self.lora_A.weight, std=init_scale)
#             nn.init.zeros_(self.lora_B.weight)
#         elif init_option == "lora":
#             # 标准LoRA初始化
#             nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
#             nn.init.zeros_(self.lora_B.weight)
#         else:
#             raise ValueError(f"未知初始化方案: {init_option}")
        
#         # Dropout和领域缩放因子
#         self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#         self.domain_scaler = nn.Parameter(torch.tensor(1.0))
        
#         # 前一个领域的权重（用于正交约束）
#         self.register_buffer('prev_domain_weight', None)
        
#         # 当前激活秩
#         self.current_rank = max_rank
        
#         # 设备占位符（用于安全剪枝）
#         self.device_placeholder = torch.empty(0)

#     def get_active_components(self):
#         """获取激活的组件和当前秩"""
#         # 使用Gumbel-Softmax近似离散采样
#         if self.training:
#             logits = self.rank_scores / self.temp
#             probs = torch.sigmoid(logits)
            
#             # ST估计器：前向传播使用硬掩码，反向传播使用软概率
#             hard_mask = (probs > 0.5).float()
#             soft_mask = probs
#             mask = hard_mask - soft_mask.detach() + soft_mask
#             active_indices = mask > 0.5
#         else:
#             # 推理时使用硬掩码
#             probs = torch.sigmoid(self.rank_scores)
#             active_indices = probs > 0.5
#             mask = active_indices.float()
        
#         active_count = torch.sum(active_indices).item()
        
#         # 确保激活秩不低于最小值
#         if active_count < self.min_rank:
#             # 选择top-k个最高概率的维度
#             _, top_indices = torch.topk(probs, self.min_rank)
#             active_indices = torch.zeros_like(active_indices, dtype=torch.bool)
#             active_indices[top_indices] = True
            
#             if self.training:
#                 # 训练时创建混合掩码
#                 mask = torch.zeros_like(mask)
#                 mask[top_indices] = probs[top_indices]
#             else:
#                 mask = active_indices.float()
            
#             active_count = self.min_rank
        
#         current_rank = min(active_count, self.max_rank)
#         return mask, active_indices, current_rank

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """前向传播，支持动态秩调整"""
#         if x.dim() != 3:
#             raise ValueError(f"预期3D输入 (batch, seq, features), 实际维度 {x.dim()}D")
            
#         mask, active_indices, current_rank = self.get_active_components()
#         self.current_rank = current_rank
        
#         # 计算动态缩放因子 (α/r)
#         scaling = self.lora_alpha / max(current_rank, 1) * self.domain_scaler
        
#         if current_rank == 0:
#             return torch.zeros_like(x)
        
#         if self.training:
#             # 训练时使用完整矩阵但应用掩码
#             A_weight = self.lora_A.weight * mask.view(-1, 1)
#             B_weight = self.lora_B.weight * mask.view(1, -1)
            
#             x_drop = self.lora_dropout(x)
#             output = F.linear(F.linear(x_drop, A_weight), B_weight)
#         else:
#             # 推理时仅使用激活部分
#             if not active_indices.any():
#                 return torch.zeros_like(x)
                
#             A_active = self.lora_A.weight[:, active_indices]
#             B_active = self.lora_B.weight[active_indices, :]
            
#             output = F.linear(F.linear(x, A_active), B_active)
        
#         return output * scaling

#     def regularization_loss(self) -> torch.Tensor:
#         """计算正则化损失，包含稀疏性和领域正交约束"""
#         if self.current_rank <= 0:
#             return torch.tensor(0.0, device=self.device_placeholder.device)
        
#         # 稀疏性正则化 (L1)
#         probs = torch.sigmoid(self.rank_scores)
#         sparse_loss = self.reg_alpha * torch.sum(probs) / self.max_rank
        
#         # 领域正交约束 (防止灾难性遗忘)
#         orth_loss = 0.0
#         if self.prev_domain_weight is not None:
#             # 鼓励当前权重与之前领域权重正交
#             cos_sim = F.cosine_similarity(
#                 self.lora_B.weight.flatten(), 
#                 self.prev_domain_weight.flatten(),
#                 dim=0
#             )
#             orth_loss = 0.1 * torch.abs(cos_sim)
        
#         return sparse_loss + orth_loss

#     def prune_parameters(self):
#         """剪枝未使用的秩维度"""
#         # 获取当前设备
#         device = self.lora_A.weight.device
        
#         # 确定激活维度
#         probs = torch.sigmoid(self.rank_scores)
#         active_indices = probs > 0.5
#         active_count = torch.sum(active_indices).item()
        
#         if active_count < self.min_rank:
#             _, top_indices = torch.topk(probs, self.min_rank)
#             active_indices = torch.zeros_like(active_indices, dtype=torch.bool)
#             active_indices[top_indices] = True
#             active_count = self.min_rank
        
#         if active_count == 0:
#             print("警告：没有激活的维度，跳过剪枝")
#             return
        
#         # 创建新的精简线性层
#         new_lora_A = nn.Linear(self.n_embd, active_count, bias=False).to(device)
#         new_lora_B = nn.Linear(active_count, self.n_embd, bias=False).to(device)
        
#         # 复制激活的权重
#         with torch.no_grad():
#             new_lora_A.weight.copy_(self.lora_A.weight[:, active_indices])
#             new_lora_B.weight.copy_(self.lora_B.weight[active_indices, :])
        
#         # 替换模块并更新参数
#         self.lora_A = new_lora_A
#         self.lora_B = new_lora_B
#         self.max_rank = active_count
        
#         # 重置秩选择参数
#         self.rank_scores = nn.Parameter(torch.ones(active_count, device=device))
#         nn.init.normal_(self.rank_scores, mean=0.0, std=0.01)

#     def clip_gradients(self, max_norm: float = 1.0):
#         """梯度裁剪"""
#         # 对秩分数进行裁剪
#         if self.rank_scores.grad is not None:
#             torch.nn.utils.clip_grad_norm_(self.rank_scores, max_norm)
        
#         # 对权重矩阵进行裁剪
#         if self.lora_A.weight.grad is not None:
#             torch.nn.utils.clip_grad_norm_(self.lora_A.parameters(), max_norm)
#         if self.lora_B.weight.grad is not None:
#             torch.nn.utils.clip_grad_norm_(self.lora_B.parameters(), max_norm)

#     def set_previous_domain_weight(self, weight: torch.Tensor):
#         """设置前一个领域的权重（用于正交约束）"""
#         self.register_buffer('prev_domain_weight', weight.detach().clone())

#     def get_effective_rank(self) -> int:
#         """获取当前有效秩"""
#         with torch.no_grad():
#             return torch.sum(torch.sigmoid(self.rank_scores) > 0.5).item()

#     # def extra_repr(self) -> str:
#     #     return f"domain={self.domain_id}, max_rank={self.max_rank}, min_rank={self.min_rank}"

class Adapter_lora(nn.Module): 
    def __init__(self,  
                 config=None, 
                 d_model=None,
                 d_model_2=None, 
                 bottleneck=None, 
                 dropout=0., 
                 init_option="bert", 
                 adapter_scalar="1.0", 
                 adapter_layernorm_option="in", 
                 max_rank: int = 64, 
                 min_rank: int = 4, 
                 init_scale: float = 0.02, 
                 temp: float = 2, 
                 reg_alpha: float = 0.1,
                 lora_alpha: float = 16.0): 
        super().__init__() 
        self.n_embd = config.d_model if d_model is None else d_model
        self.n_embd_2 = config.d_model if d_model_2 is None else d_model_2 
        self.max_rank = max_rank 
        self.min_rank = min_rank 
        temp = max(float(temp), 1e-2)
        self.temp = temp 
        self.lora_alpha = lora_alpha  # LoRA缩放因子
         
        # 可学习参数 - 更保守的初始化
        self.rank_scores = nn.Parameter(torch.ones(max_rank))  
        with torch.no_grad():
            self.rank_scores.data.normal_(mean=0.0, std=0.01)
        self.register_buffer('active_mask', torch.ones(max_rank, dtype=torch.bool)) 
        self.random_orth = True
        # LoRA矩阵定义
        self.lora_A = nn.Linear(self.n_embd, max_rank, bias=False)
        self.lora_B = nn.Linear(max_rank, self.n_embd_2, bias=False)
        
        if self.random_orth:
            random_matrix = torch.rand(self.n_embd, max_rank)
            q, r = torch.linalg.qr(random_matrix)
            with torch.no_grad():
                self.lora_A.weight.copy_(q.T)
            scaling_factor = 1.  # You can adjust this value if needed
            self.lora_A.weight.data *= scaling_factor
        else:
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.zeros_(self.lora_B.weight)
        else:
            raise NotImplementedError
        
        # 动态正则化参数 
        self.reg_alpha = reg_alpha 
        self.current_rank = max_rank
        
        # 添加dropout防止过拟合
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def get_active_components(self): 
        """获取激活的组件，确保mask和active_indices完全一致"""
        if self.training: 
            # 使用straight-through estimator避免梯度问题
            logits = self.rank_scores / self.temp 
            soft_mask = torch.sigmoid(logits)
            hard_mask = (soft_mask > 0.5).float()
            # 正确的Straight-through trick: forward用hard，backward用soft
            base_mask = hard_mask.detach() + soft_mask - soft_mask.detach()
            base_active_indices = (hard_mask > 0.5)
        else: 
            sigmoid_scores = torch.sigmoid(self.rank_scores)
            base_active_indices = (sigmoid_scores > 0.5)
            base_mask = base_active_indices.float()
         
        # 计算当前激活的秩
        current_active_count = torch.sum(base_active_indices).item()
        
        # 关键修复：如果激活太少，需要同时更新mask和active_indices
        if current_active_count < self.min_rank:
            # 选择top-k个最高分数的维度
            _, top_indices = torch.topk(self.rank_scores, self.min_rank)
            
            # 创建新的active_indices（布尔张量）
            final_active_indices = torch.zeros_like(base_active_indices, dtype=torch.bool)
            final_active_indices[top_indices] = True
            
            if self.training:
                # 训练时：为top-k维度创建新的mask
                final_mask = torch.zeros_like(base_mask)
                for idx_t in top_indices:
                    idx = int(idx_t.item())
                    final_mask[idx] = 1.0 - soft_mask[idx].detach() + soft_mask[idx]
            else:
                # 推理时：直接使用hard mask
                final_mask = final_active_indices.float()
            
            final_active_count = self.min_rank
        else:
            # 如果激活数量足够，直接使用base结果
            final_mask = base_mask
            final_active_indices = base_active_indices
            final_active_count = current_active_count
        
        # 确保active_rank在合理范围内
        final_active_rank = min(final_active_count, self.max_rank)
        
        # 验证一致性（调试用，可以删除）
        assert torch.sum(final_active_indices).item() == final_active_rank, \
            f"Inconsistency: active_indices sum {torch.sum(final_active_indices).item()} != active_rank {final_active_rank}"
        self.active_mask = final_mask
        return final_mask, final_active_indices, torch.tensor(final_active_rank, device=self.rank_scores.device)

    def forward(self, x): 
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq, features), got {x.dim()}D")
            
        mask, active_indices, current_rank = self.get_active_components()
        self.current_rank = current_rank.item()
        
        # 动态缩放因子
        if self.current_rank > 0:
            dynamic_scaling = self.lora_alpha / self.max_rank
        else:
            return torch.zeros_like(x)
         
        if self.training:
            # 训练时使用mask加权
            A_weighted = self.lora_A.weight * mask.unsqueeze(1)
            B_weighted = self.lora_B.weight * mask.unsqueeze(0)
            
            # 添加dropout
            x_dropout = self.lora_dropout(x)
            x_dropout = F.normalize(x_dropout, p=2, dim=1)
            # A_weighted = F.normalize(A_weighted, p=2, dim=1)
            # B_weighted = F.normalize(B_weighted, p=2, dim=1)
            hidden = F.linear(x_dropout, A_weighted)
            output = F.linear(hidden, B_weighted)     # [batch, seq, n_embd]
            
        else:
            # 推理时只使用激活的维度
            if active_indices.any():
                A_active = self.lora_A.weight[active_indices, :]
                B_active = self.lora_B.weight[:, active_indices]
                
                x = self.lora_dropout(x)
                x = F.normalize(x, p=2, dim=1)
                # A_active = F.normalize(A_active, p=2, dim=1)
                # B_active = F.normalize(B_active, p=2, dim=1)
                hidden = F.linear(x, A_active)
                output = F.linear(hidden, B_active)
            else:
                output = torch.zeros_like(x)
        
        # 应用缩放因子
        return output * dynamic_scaling
 
    def regularization_loss(self): 
        if self.current_rank <= 0:
            return torch.tensor(0.0, device=self.rank_scores.device)    
        # # L1正则化鼓励稀疏性
        # 使用sigmoid激活的和作为复杂度惩罚
        active_ratio = torch.sum(torch.sigmoid(self.rank_scores)) / self.max_rank
        reg_loss = self.reg_alpha * active_ratio
        return reg_loss
 
    @torch.no_grad() 
    def prune_parameters(self): 
        # 修复：安全的参数剪枝
        
        mask, self.active_mask, current_rank = self.get_active_components()
        active_count = torch.sum(self.active_mask).item()
           
        if active_count == 0:
            print("警告：没有激活的维度，跳过剪枝")
            return
            
        # 创建新的线性层
        new_lora_A = nn.Linear(self.n_embd, active_count, bias=False).to(self.lora_A.weight.device)
        new_lora_B = nn.Linear(active_count, self.n_embd, bias=False).to(self.lora_B.weight.device)
        self.max_rank = active_count
        # 复制激活的权重
        new_lora_A.weight.data = self.lora_A.weight[self.active_mask, :].clone()
        new_lora_B.weight.data = self.lora_B.weight[:, self.active_mask].clone()
        
        # old_W = self.lora_B.weight @ self.lora_A.weight   
        # 替换模块
        self.lora_A = new_lora_A
        self.lora_B = new_lora_B
         
        # 更新参数
        # self.max_rank = active_count
        self.rank_scores = nn.Parameter(torch.ones(self.max_rank, device=self.rank_scores.device))
        self.register_buffer('active_mask', torch.ones(self.max_rank, dtype=torch.bool))
      
    # @torch.no_grad()
    # def prune_parameters(self): 
    #     """安全剪枝：保证剪枝前后 forward 输出一致"""

    #     # 获取当前mask和active indices
    #     mask, active_indices, current_rank = self.get_active_components()
    #     active_count = torch.sum(active_indices).item()
        
    #     if active_count == 0:
    #         print("警告：没有激活的维度，跳过剪枝")
    #         return
        
    #     # ----------- 关键修改点 -----------
    #     # 1. 先应用 mask + normalize，再提取子矩阵
    #     # A_weighted = self.lora_A.weight * mask.unsqueeze(1)
    #     # B_weighted = self.lora_B.weight * mask.unsqueeze(0)

    #     # A_weighted = F.normalize(A_weighted, p=2, dim=1)
    #     # B_weighted = F.normalize(B_weighted, p=2, dim=1)

    #     # 2. 只保留 active 的行/列
    #     A_active = A_weighted[active_indices, :].clone()
    #     B_active = B_weighted[:, active_indices].clone()
    #     # --------------------------------

    #     # 创建新的线性层
    #     new_lora_A = nn.Linear(self.n_embd, active_count, bias=False).to(self.lora_A.weight.device)
    #     new_lora_B = nn.Linear(active_count, self.n_embd, bias=False).to(self.lora_B.weight.device)

    #     # 拷贝权重
    #     new_lora_A.weight.data = A_active
    #     new_lora_B.weight.data = B_active

    #     # 替换模块
    #     self.lora_A = new_lora_A
    #     self.lora_B = new_lora_B

    #     # 更新 rank_scores，只保留被选中的部分
    #     self.rank_scores = nn.Parameter(self.rank_scores[active_indices].clone())

    #     # 更新 active_mask
    #     self.register_buffer('active_mask', torch.ones(active_count, dtype=torch.bool))

    #     # 更新 max_rank
    #     self.max_rank = active_count
       
    def clip_gradients(self, max_norm=1.0): 
        # 梯度裁剪 
        if self.rank_scores.grad is not None:
            torch.nn.utils.clip_grad_norm_(self.rank_scores, max_norm)
            
    def get_effective_rank(self):
        """获取当前有效秩"""
        with torch.no_grad():
            return torch.sum(torch.sigmoid(self.rank_scores) > 0.5).item()

class Attention_lora(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., msa = [0,0,0]):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.ffn_option = 'parallel'
        self.msa = msa


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()


    def forward(self, x, adapt=None, prompt = None, rank_prompt = None, block_weight = None):
        B, N, C = x.shape

        q = self.q_proj(x) # 64 197 768
        k = self.k_proj(x)
        v = self.v_proj(x)

        if adapt is not None:
            if block_weight is not None:
                block_weight = block_weight
            else:
                block_weight = torch.ones(4).cuda()
            if self.msa[0] == 1:
                adapt_x = adapt[0](x)
                q += block_weight[0] * adapt_x
            if self.msa[1] == 1:
                adapt_x = adapt[1](x)
                k += block_weight[1] * adapt_x
            if self.msa[2] == 1:
                adapt_x = adapt[2](x)
                v += block_weight[2] * adapt_x


        k = self._shape(k, -1, B).view(B * self.num_heads, -1, self.head_dim) # 64 197 768 -> 64 12 197 64 -> 768 197 64
        v = self._shape(v, -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)


        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)
        if adapt is not None:
            if self.msa[3] == 1:
                adapt_x = adapt[3](attn_output)
                attn_output = attn_output + block_weight[3] * adapt_x #  

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        super().__init__()
        self.config = config
        self.msa_adapt = True
        self.norm1 = norm_layer(dim)
        self.attn = Attention_lora(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, msa = config.msa)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)
        self.msa = config.msa



    # prompt and rank_prmopt can be considerred as potential future improvements by levergaing additional prompt information, but is not implemented in this work
    def forward(self, x, adapt=None, prompt=None, rank_prompt=None, block_weight=None):
        if self.msa_adapt:
            x = x + self.drop_path(
                self.attn(self.norm1(x), adapt, prompt, rank_prompt, block_weight))
            residual = x
            hidden = self.fc1(self.norm2(x))

            # fc1 lora
            if adapt is not None and len(adapt) > 4 and self.msa[4] == 1:
                if block_weight is None:
                    block_weight = torch.ones(len(adapt)).to(hidden.device)
                adapt_x = adapt[4](self.norm2(x))
                hidden = hidden + adapt_x

            hidden = self.act(hidden)
            hidden = self.mlp_drop(hidden)

            out = self.fc2(hidden)

            # fc2 lora
            if adapt is not None and len(adapt) > 5 and self.msa[5] == 1:
                if block_weight is None:
                    block_weight = torch.ones(len(adapt)).to(out.device)
                adapt_x = adapt[5](hidden)
                out = out + adapt_x

            out = self.mlp_drop(out)
            x = residual + self.drop_path(out)

        return x



class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None):
        super().__init__()

        self.tuning_config = tuning_config
        if self.tuning_config.ffn_adapt:
            print("I'm using ViT with adapters.")
        else:
            print("I'm using ViT without adapters.")
            self.maskout_block = []
        self.adapt_msa = True
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.msa_adapt = self.tuning_config.msa_adapt
        self.use_distillation = self.tuning_config.use_distillation
        self.use_block_weight = self.tuning_config.use_block_weight

        if self.msa_adapt:
            self.msa = self.tuning_config.msa
        self.general_pos = self.tuning_config.general_pos
        self.specfic_pos = self.tuning_config.specfic_pos

        self.adapt_pos = self.general_pos+ self.specfic_pos
        self.adapt_pos = sorted(self.adapt_pos)


        if self.use_distillation:
            self.old_adapter_list = nn.ModuleList()

        if self.use_block_weight:
            self.block_weight_list = []
            self.block_weight = nn.Parameter(torch.randn(4, len(self.specfic_pos)))
            nn.init.uniform_(self.block_weight, .5, 1.5)


        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        ######### MAE begins ############
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        ######## Adapter begins #########
        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0, tuning_config.vpt_num
            # properly registered
            self.embeddings = nn.ParameterList(  # batch, num_prompt, embed_dim
                [nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim)) for _ in
                 range(depth)])
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)

        self.config = tuning_config
        self._device = tuning_config._device
        self.adapter_list = []
        self.adapter_pos_list = []
        self.cur_adapter = nn.ModuleList()
        if self.msa_adapt:
            self.get_new_adapter_initial_msa()

    def init_weights(self, mode=''):
        raise NotImplementedError()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist           

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        
        for i in range(len(self.cur_adapter)):
            self.cur_adapter[i].requires_grad = True


    def get_new_adapter_initial_msa(self):
        config = self.config
        if config.ffn_adapt:
            for i in range(len(self.adapt_pos)):
                temp_adapter = nn.ModuleList()
                for j in range(len(self.msa)):
                    if self.msa[j] ==1 and j not in [4,5]:
                        adapter = Adapter_lora(self.config, dropout=0.0, bottleneck=config.ffn_num,
                                                init_option=config.ffn_adapter_init_option,
                                                adapter_scalar=config.ffn_adapter_scalar,
                                                adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                                ).to(self._device)
                    elif self.msa[j] ==1 and j==4:
                        adapter = Adapter_lora(self.config, d_model_2=3072,dropout=0.0, bottleneck=config.ffn_num,
                                                init_option=config.ffn_adapter_init_option,
                                                adapter_scalar=config.ffn_adapter_scalar,
                                                adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                                ).to(self._device)
                    elif self.msa[j] ==1 and j==5:
                        adapter = Adapter_lora(self.config, d_model=3072, d_model_2=self.config.d_model,dropout=0.0, bottleneck=config.ffn_num,
                                                init_option=config.ffn_adapter_init_option,
                                                adapter_scalar=config.ffn_adapter_scalar,
                                                adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                                ).to(self._device)
                    else:
                        adapter = nn.Identity()
                    temp_adapter.append(adapter)

                self.cur_adapter.append(temp_adapter)
            self.cur_adapter.requires_grad_(True)

        else:
            print("====Not use adapter===")

    def get_new_adapter_msa(self):
        config = self.config

        if config.ffn_adapt:
            for i in range(len(self.specfic_pos)):
                pos = self.adapt_pos.index(self.specfic_pos[i])
                temp_adapter = nn.ModuleList()
                for j in range(len(self.msa)):
                    if self.msa[j] ==1 and j not in [4,5]:
                        adapter = Adapter_lora(self.config, dropout=0.0, bottleneck=config.ffn_num,
                                                init_option=config.ffn_adapter_init_option,
                                                adapter_scalar=config.ffn_adapter_scalar,
                                                adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                                ).to(self._device)
                        adapter.requires_grad_(True)
                    elif self.msa[j] ==1 and j==4:
                        adapter = Adapter_lora(self.config, d_model_2=3072,dropout=0.0, bottleneck=config.ffn_num,
                                                init_option=config.ffn_adapter_init_option,
                                                adapter_scalar=config.ffn_adapter_scalar,
                                                adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                                ).to(self._device)
                        adapter.requires_grad_(True)
                    elif self.msa[j] ==1 and j==5:
                        adapter = Adapter_lora(self.config, d_model=3072, d_model_2=self.config.d_model,dropout=0.0, bottleneck=config.ffn_num,
                                                init_option=config.ffn_adapter_init_option,
                                                adapter_scalar=config.ffn_adapter_scalar,
                                                adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                                ).to(self._device)
                        adapter.requires_grad_(True)
                    else:
                        adapter = nn.Identity()
                    temp_adapter.append(adapter)
                self.cur_adapter[pos] = temp_adapter

            # if len(self.specfic_pos) < 12:
            #     self.cur_adapter.requires_grad_(True)

            #     for i in self.adapt_pos:
            #         if i in self.general_pos:
            #             pos = self.adapt_pos.index(i)
            #             for j in range(len(self.msa)):
            #                 if self.msa[j] == 1:
            #                     self.cur_adapter[pos][j].lora_B.requires_grad_(False)
        else:
            print("====Not use adapter===")


    def add_adapter_to_list(self):
        temp_adapter = []
        for i in range(len(self.specfic_pos)):
            temp_pos = self.adapt_pos.index(self.specfic_pos[i])
            temp_adapter.append(copy.deepcopy(self.cur_adapter[temp_pos].requires_grad_(False)))
        self.adapter_list.append(temp_adapter)

        if self.use_block_weight:
            self.block_weight_old = copy.deepcopy(self.block_weight)
            self.block_weight_list.append(self.block_weight_old.requires_grad_(False))
            self.block_weight = nn.Parameter(torch.randn(4, len(self.specfic_pos)))
            nn.init.uniform_(self.block_weight, .5, 1.5)
            print(self.block_weight_list)


        self.adapter_pos_list.append(self.adapt_pos)

        if self.use_distillation:
            self.old_adapter_list.append(copy.deepcopy(self.cur_adapter).requires_grad_(False))
        if self.msa_adapt:
            self.get_new_adapter_msa()

    def forward_train(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            rank_prompt = None
            prompt = None

            if self.config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)

            if self.config.ffn_adapt:
                if idx in self.adapt_pos:
                    pos = self.adapt_pos.index(idx)
                    block_weight = None
                    if self.use_block_weight and idx in self.specfic_pos:
                        pos_spec = self.specfic_pos.index(idx)
                        x = blk(x, self.cur_adapter[pos], prompt, rank_prompt,
                                block_weight=self.block_weight[:, pos_spec])
                    else:
                        x = blk(x, self.cur_adapter[pos], prompt, rank_prompt, block_weight=None)
                else:
                    x = blk(x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)
            else:
                x = blk(x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)
            if self.config.vpt_on:
                x = x[:, self.config.vpt_num:, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward_test(self, x, use_init_ptm=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x_init = self.pos_drop(x)

        features = []

        if use_init_ptm:
            x = copy.deepcopy(x_init)
            x = self.blocks(x)
            x = self.norm(x)
            features.append(x)
        if self.config.ffn_adapt:
            for i in range(len(self.adapter_list)):
                x = copy.deepcopy(x_init)
                for j in range(len(self.blocks)):

                    rank_prompt = None
                    prompt = None

                    if j in self.adapt_pos:
                        if j in self.general_pos:
                            pos = self.adapt_pos.index(j)
                            adapt = self.cur_adapter[pos]
                        else:
                            pos = self.specfic_pos.index(j)
                            adapt = self.adapter_list[i][pos]

                        if self.use_block_weight and j in self.specfic_pos:
                            pos_spec = self.specfic_pos.index(j)
                            block_weight = self.block_weight_list[i][:, pos_spec]
                        else:
                            block_weight = None
                        x = self.blocks[j](x, adapt, prompt, rank_prompt, block_weight)

                    else:
                        x = self.blocks[j](x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)

                x = self.norm(x)
                features.append(x)

            x = copy.deepcopy(x_init)
            for i in range(len(self.blocks)):

                rank_prompt = None
                prompt = None

                if i in self.adapt_pos:
                    pos = self.adapt_pos.index(i)
                    adapt = self.cur_adapter[pos]
                    if self.use_block_weight and i in self.specfic_pos:
                        pos_spec = self.specfic_pos.index(i)
                        block_weight = self.block_weight[:, pos_spec]
                    else:
                        block_weight = None
                    x = self.blocks[i](x, adapt, prompt, rank_prompt, block_weight)
                else:
                    x = self.blocks[i](x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)
            x = self.norm(x)
            features.append(x)

        return features

    def forward(self, x, test=False, use_init_ptm=False):
        if not test:
            output = self.forward_train(x)
            with torch.no_grad():
                pre_output = self.forward_test(x, use_init_ptm)
                outputs = torch.Tensor().to(pre_output[0].device)
                for x in pre_output:
                    cls = x[:, 0, :]
                    outputs = torch.cat((
                        outputs,
                        cls
                    ), dim=1)
            return output

        else:
            features = self.forward_test(x, use_init_ptm)
            output = torch.Tensor().to(features[0].device)
            for x in features:
                cls = x[:, 0, :]
                output = torch.cat((
                    output,
                    cls
                ), dim=1)
            return output

    def forward_proto(self, x, adapt_index):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x_init = self.pos_drop(x)

        # the init_PTM's feature
        if adapt_index == -1:
            x = copy.deepcopy(x_init)
            x = self.blocks(x)
            x = self.norm(x)
            output = x[:, 0, :]
            return output

        i = adapt_index
        x = copy.deepcopy(x_init)
        if self.config.ffn_adapt:
            if i < len(self.adapter_list):
                for j in range(len(self.blocks)):

                    rank_prompt = None
                    prompt = None

                    if j in self.adapt_pos:
                        if j in self.general_pos:
                            pos = self.adapt_pos.index(j)
                            adapt = self.cur_adapter[pos]
                        else:
                            pos = self.specfic_pos.index(j)
                            adapt = self.adapter_list[i][pos]
                        if self.use_block_weight and j in self.specfic_pos:
                            pos_spec = self.specfic_pos.index(j)
                            block_weight = self.block_weight_list[i][:, pos_spec]
                        else:
                            block_weight = None
                        x = self.blocks[j](x, adapt, prompt, rank_prompt, block_weight)

                    else:
                        x = self.blocks[j](x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)
            else:
                for j in range(len(self.blocks)):
                    rank_prompt = None
                    prompt = None

                    if j in self.adapt_pos:
                        pos = self.adapt_pos.index(j)
                        adapt = self.cur_adapter[pos]
                        if self.use_block_weight and j in self.specfic_pos:
                            pos_spec = self.specfic_pos.index(j)
                            block_weight = self.block_weight[:, pos_spec]
                        else:
                            block_weight = None

                        x = self.blocks[j](x, adapt, prompt, rank_prompt, block_weight)
                    else:
                        x = self.blocks[j](x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)
        else:
            for j in range(len(self.blocks)):
                rank_prompt = None
                prompt = None

                x = self.blocks[j](x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)

        x = self.norm(x)
        output = x[:, 0, :]

        return output

    def forward_general_cls(self, x, t_idx):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x_teacher = copy.deepcopy(x)

        for j in self.general_pos:
            pos = self.adapt_pos.index(j)
            adapt = self.cur_adapter[pos]
            x = self.blocks[j](x, adapt)

        x = self.norm(x)
        output_new = x[:, 0, :]



        for j in self.general_pos:
            pos = self.adapt_pos.index(j)
            adapt = self.old_adapter_list[t_idx-1][pos]
            x_teacher = self.blocks[j](x_teacher, adapt)
        x_teacher = self.norm(x_teacher)
        output_teacher= x_teacher[:, 0, :]

        return output_new, output_teacher



def vit_base_patch16_224_ours(pretrained=False, **kwargs):
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    checkpoint_model=timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False

    if not model.msa_adapt:
        for adapter_temp in model.cur_adapter:
            #for adapter in adapter_temp:
            for param in adapter_temp.lora_B.parameters():
                param.requires_grad = False
    else:
        for i in model.adapt_pos:
            #if i in model.general_pos:
            if i in model.general_pos:
                pos = model.adapt_pos.index(i)
                for j in range(len(model.msa)):
                    if model.msa[j] == 1:
                    #for adapter in adapter_temp:
                        for param in model.cur_adapter[pos][j].lora_B.parameters():
                            param.requires_grad = False
    #
    return model

def vit_base_patch16_224_in21k_ours(pretrained=False, **kwargs):
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    checkpoint_model=timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False


    # if not model.msa_adapt:
    #     for adapter_temp in model.cur_adapter:
    #         #for adapter in adapter_temp:
    #         for param in adapter_temp.lora_B.parameters():
    #             param.requires_grad = False
    # else:
    #     for i in model.adapt_pos:
    #         #if i in model.general_pos:
    #         if i in model.general_pos:
    #             pos = model.adapt_pos.index(i)
    #             for j in range(len(model.msa)):
    #                 if model.msa[j] == 1:
    #                 #for adapter in adapter_temp:
    #                     for param in model.cur_adapter[pos][j].lora_B.parameters():
    #                         param.requires_grad = False

    return model


def load_npz_to_state_dict(filename):
    # Load the .npz file
    with np.load(filename, allow_pickle=True) as data:
        state_dict = {}
        for key in data.keys():
            state_dict[key] = torch.from_numpy(data[key])
    return state_dict

def compute_column_importance(matrix):
    """
    Compute importance of each column based on SVD and scale to range (0, 1).
    """
    U, S, Vt = torch.linalg.svd(matrix.T, full_matrices=False)
    importance_scores = torch.sum(torch.abs(U * S), dim=1)
    scaled_scores = (importance_scores - torch.min(importance_scores)) / (torch.max(importance_scores) - torch.min(importance_scores))
    epsilon = 1e-10
    scaled_scores = torch.maximum(scaled_scores, torch.tensor(epsilon))
    return scaled_scores


def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))
