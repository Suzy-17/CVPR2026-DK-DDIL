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
import random
import os 
num_workers = 4


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
        
        self.anti_prototype_loss = DomainContrastiveLoss()
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

        with torch.no_grad():
            # replace proto for each adapter in the current task
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
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    self.class_proto = torch.cat((self.class_proto, proto.unsqueeze(0).to(self._device)), dim=0)
                    if self.use_init_ptm:
                        model.fc.weight.data[class_index, (index+1)*self._network.out_dim:(index+2)*self._network.out_dim] = proto
                    else:
                        model.fc.weight.data[class_index, index*self._network.out_dim:(index+1)*self._network.out_dim] = proto
        return
    
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
            if is_best:
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
            
            # 定期进行LoRA剪枝（可选）
            if (epoch + 1) % getattr(self.args, 'prune_interval', 20) == 0:
                self._prune_lora_adapters(epoch)
        
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
                anti_prototype_loss = self.anti_prototype_loss(output['features'], aux_targets, self.cur_class_proto, self.class_proto)
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
        # for lora in lora_adapters:
        #     if hasattr(lora, 'prune_parameters'):
        #         lora.prune_parameters()
        #     if hasattr(lora, 'prune_parameters'):
        #         lora.prune_parameters()
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
        if not getattr(self.args, 'enable_pruning', False):
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
                effective_rank = lora.get_effective_rank()
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
