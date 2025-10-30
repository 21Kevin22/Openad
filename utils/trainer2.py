import numpy as np
import torch
import torch.nn as nn
import os
from os.path import join as opj
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import *
from utils.eval import evaluation
from time import time
from torch.amp import autocast, GradScaler
import wandb


class Trainer(object):
    def __init__(self, cfg, running):
        super().__init__()
        self.cfg = cfg
        self.work_dir = self.cfg.work_dir
        try:
                # If cfg has a method to convert to dict (common in config libraries)
                # Adjust '.to_dict()' or '.dump()' based on your library (gorilla.config might use ._cfg_dict)
                if hasattr(cfg, '_cfg_dict'):
                    config_dict = cfg._cfg_dict
                elif hasattr(cfg, 'to_dict'):
                    config_dict = cfg.to_dict()
                elif hasattr(cfg, 'dump'):
                    # gorilla.config dump() returns a string, parse it back (less ideal)
                    import json
                    config_dict = json.loads(cfg.dump())
                else:
                    # Fallback: Manually create or filter (might need adjustment)
                    config_dict = {k: v for k, v in vars(cfg).items() if isinstance(v, (str, int, float, bool, list, dict, tuple))}
                    logger.cprint("[WARN] Using fallback method to create config_dict for wandb.")

        except Exception as e:
                logger.cprint(f"[WARN] Could not automatically convert cfg to dict for wandb: {e}. Using None.")
                config_dict = None # Log minimal config if conversion fails

            # Initialize wandb with the serializable dictionary
        wandb.init(
                project="Open-Vocabulary-Affordance", 
                config=config_dict # Pass the cleaned dictionary
            )
        self.writer = SummaryWriter(self.work_dir)
        self.logger = running['logger']
        self.model = running["model"]

        wandb.watch(self.model, log="all", log_freq=100)

        self.dataset_dict = running["dataset_dict"]
        self.loader_dict = running["loader_dict"]
        self.train_loader = self.loader_dict.get("train_loader", None)
        self.val_loader = self.loader_dict.get("val_loader", None)
        self.loss = running["loss"]
        self.optim_dict = running["optim_dict"]
        self.optimizer = self.optim_dict.get("optimizer", None)
        self.scheduler = self.optim_dict.get("scheduler", None)
        self.epoch = 0
        self.best_val_mIoU = 0.0
        self.bn_momentum = self.cfg.training_cfg.get('bn_momentum', None)
        self.train_affordance = cfg.training_cfg.train_affordance
        self.val_affordance = cfg.training_cfg.val_affordance
        
        self.scaler = GradScaler('cuda') 
        return

    def train(self):
        train_loss = 0.0
        count = 0.0
        
        # ★ 解決策 (L46): train モードにし、勾配計算を明示的に有効化
        self.model.train()
        torch.set_grad_enabled(True) # ★ 勾配計算をオンにする
        # ★ 解決策 完了
        
        num_batches = len(self.train_loader)
        start = time()
        self.logger.cprint("Epoch(%d) begin training........" % self.epoch)
        
        for data, label in tqdm(self.train_loader, total=len(self.train_loader), smoothing=0.9):
            data, label = data.float().cuda(), label.float().cuda()
            
            # データは既に [B, C, N] 形式なので、そのまま使用
            # ラベルを [B, N] の long 型に正規化

            label = label.long().contiguous()  # 既に (1, 4096) の形状
            
            batch_size = data.size()[0]
            num_point_data = data.size()[2] # (例: 4096)
            
            # (注: zero_grad() は scaler.step() の後が一般的だが、ここではAMPの有無によらず動作させるため先頭に置く)
            self.optimizer.zero_grad() 
            
            # ★ 改善 3: 混合精度 (autocast) の適用
            # (FutureWarning 修正: 'cuda' を位置引数として渡す)
            with autocast('cuda'):
                afford_pred = self.model(data, self.train_affordance)
                afford_pred = afford_pred.contiguous()

                # --- ★ エラー修正 (L59) ---
                # モデルが 2D ([B*M, C] または [M, C]) を返した場合、
                # 3D ([B, M, C]) に正規化します。
                B_data = data.size(0)
                
                if afford_pred.dim() == 2:
                    M_flat, C_out_flat = afford_pred.shape
                    N = data.shape[2]  # define N early to avoid NameError
                    M = M_flat // B_data 
                    if M_flat % B_data != 0:
                        if B_data == 1 and M_flat > C_out_flat: 
                            M = M_flat
                        else:
                             raise ValueError(f"Cannot reshape afford_pred [B*M, C] (shape {afford_pred.shape}) with B={B_data}")
                    print(f"[DEBUG] Reshaping 2D afford_pred {afford_pred.shape} -> 3D [{B_data}, {M}, {C_out_flat}]")
                    afford_pred = afford_pred.view(B_data, M, C_out_flat) 

                if afford_pred.dim() == 3:
                    # ✅ 修正点: ここに permute を追加
                    # モデル出力 [B, C, N] (例: [1, 19, 4096]) を
                    # [B, N, C] (例: [1, 4096, 19]) に入れ替える
                    afford_pred = afford_pred.permute(0, 2, 1).contiguous()
                    
                    B_out, M, C_out = afford_pred.shape # (B_out=1, M=4096, C_out=19)
                else:
                    raise ValueError(f"Unexpected afford_pred dim after normalization: {afford_pred.dim()}")
                # --- ★ エラー修正完了 ---

                if label.ndim == 3:
                    if label.shape[1] == 1:
                        label = label.squeeze(1)
                    elif label.shape[2] == 1:
                        label = label.squeeze(2)
               
                # 安全にNを取得（2次元: [B, N], 1次元: [N]）
                if label.ndim == 2:
                    N = label.shape[1] # (N=4096)
                elif label.ndim == 1:
                    N = label.shape[0]
                else:
                    raise ValueError(f"label shape不正: {label.shape}")
                
                # (M=4096, N=4096 なので、ここはスキップされるはず)
                if M != N:  # Interpolate or repeat for up/down sampling
                    if N > M:
                        afford_pred = afford_pred[:, :N, :].contiguous()
                        print(f"[INFO] Upsampled afford_pred from {M} -> {N} points")
                    else:
                        # ダウンサンプリング: 順序を保って間引き(N等分サンプリング)
                        idx = torch.linspace(0, M - 1, steps=N).long().to(afford_pred.device)
                        afford_pred = torch.index_select(afford_pred, 1, idx).contiguous()
                        print(f"[INFO] Downsampled afford_pred from {M} -> {N} points via linear indexing.")

                B_out_up, N_out_up, C_out_up = afford_pred.shape # (1, 4096, 19)

                # (★ 平坦化: loss.py が [B*N, C] と [B*N] を期待するため)
                afford_pred_flat = afford_pred.contiguous().view(B_out_up * N_out_up, C_out_up) # (4096, 19)
                label_flat = label.contiguous().view(-1) # (4096)

                if label_flat.dtype != torch.long:
                    print(f"[DEBUG] Casting label_flat from {label_flat.dtype} to torch.long")
                    label_flat = label_flat.long()
                

                if afford_pred_flat.shape[0] != label_flat.shape[0]:
                    min_flat = min(afford_pred_flat.shape[0], label_flat.shape[0])
                    print(f"[WARN] afford_pred_flat and label_flat size mismatch ({afford_pred_flat.shape[0]}, {label_flat.shape[0]}), truncating to {min_flat}")
                    afford_pred_flat = afford_pred_flat[:min_flat, :]
                    label_flat = label_flat[:min_flat]

                # loss.py (NLLLoss) は pred [N, C] (4096, 19) と target [N] (4096) を期待
                # weights は [C] (19) を期待
                # これで形状が一致する
                loss = self.loss(afford_pred_flat, label_flat)
            
                # === 混合精度対応 ===
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            count += batch_size * num_point_data
            train_loss += loss.item()

            try:
                del data, label, afford_pred, afford_pred_flat, label_flat, loss
            except Exception as e:
                # 削除に失敗しても学習は継続
                self.logger.cprint(f"[WARN] Failed to del tensors in train loop: {e}")
            
            # tqdm の表示を更新
            pbar.set_description(f"Epoch {self.epoch}, Loss: {current_loss_item:.6f}")
        
        self.scheduler.step()
        if self.bn_momentum != None:
            self.model.apply(lambda x: self.bn_momentum(x, self.epoch))
        epoch_time = time() - start
        outstr = 'Train(%d), loss: %.6f, time: %d s' % (
            self.epoch, train_loss*1.0/num_batches, epoch_time//1)
        self.writer.add_scalar('Loss', train_loss*1.0/num_batches, self.epoch)
        wandb.log({
            "epoch": self.epoch,
            "train/loss": avg_loss,
            "train/epoch_time_sec": epoch_time // 1,
            "train/learning_rate": self.optimizer.param_groups[0]['lr']
        })
        self.logger.cprint(outstr)
        self.epoch += 1
        torch.cuda.empty_cache()

    def val(self):
        self.logger.cprint('Epoch(%d) begin validating......' % (self.epoch-1))
        
        # ★ 解決策 (L151): 評価中は勾配計算を明示的に無効化
        self.model.eval() # 評価モード
        with torch.no_grad(): # ★ 勾配計算をオフにする
            mIoU = evaluation(self.logger, self.model,
                             self.val_loader, self.val_affordance)

        wandb.log({
            "epoch": self.epoch - 1, 
            "val/mIoU": mIoU
        })
        # ★ 解決策 完了

        if mIoU >= self.best_val_mIoU:
            self.best_val_mIoU = mIoU
            self.logger.cprint('Saving model......')
            self.logger.cprint('Best mIoU: %f' % self.best_val_mIoU)
            torch.save(self.model.state_dict(),
                   opj(self.work_dir, 'best_model.t7'))

            wandb.save(model_path=opj(self.work_dir, 'best_model.t7'), base_path=self.work_dir)

        torch.save(self.model.state_dict(),
                      opj(self.work_dir, 'current_model.t7'))
            

    def run(self):
        EPOCH = self.cfg.training_cfg.epoch
        workflow = self.cfg.training_cfg.workflow
        try:
            while self.epoch < EPOCH:
                for key, running_epoch in workflow.items():
                    epoch_runner = getattr(self, key)
                    for e in range(running_epoch):
                        epoch_runner()
        except Exception as e:
            self.logger.cprint(f"[ERROR] Training failed: {e}")
            wandb.log({"error": str(e)})
        finally:
            wandb.finish()