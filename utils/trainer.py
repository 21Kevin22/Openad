import numpy as np
import torch
import torch.nn as nn
import os
from os.path import join as opj
from tqdm import tqdm
from tensorboardX import SummaryWriter # Keep if using alongside wandb
from utils import * # Make sure evaluation is imported correctly if it's here
# If evaluation is separate: from utils.eval import evaluation
from time import time
from torch.amp import autocast, GradScaler
import wandb
import gorilla # Assuming gorilla.config.Config is used
import json # Needed for potential config dump loading

class Trainer(object):
    def __init__(self, cfg, running):
        super().__init__()
        self.cfg = cfg
        self.work_dir = self.cfg.work_dir
        self.logger = running['logger']
        self.writer = SummaryWriter(self.work_dir)

        # --- config を辞書に変換 ---
        config_dict = None
        try:
            if hasattr(cfg, '_cfg_dict'):
                config_dict = cfg._cfg_dict
            elif hasattr(cfg, 'to_dict'):
                config_dict = cfg.to_dict()
            elif hasattr(cfg, 'dump'):
                config_dict = json.loads(cfg.dump())
            else:
                # Fallback: 基本型のみ抽出 (ここでも module を除外)
                config_dict = {k: v for k, v in vars(cfg).items() if isinstance(v, (str, int, float, bool, list, dict, tuple, type(None)))}
                self.logger.cprint("[WARN] Using fallback method to create config_dict for wandb.")

            # --- ✅ 辞書を再帰的にクリーンアップ ---
            if config_dict is not None:
                cleaned_config_dict = clean_config_for_wandb(config_dict)
            else:
                cleaned_config_dict = None

        except Exception as e:
            self.logger.cprint(f"[WARN] Could not convert/clean cfg for wandb: {e}. Logging config as None.")
            cleaned_config_dict = None

        # --- wandb を初期化 ---
        try:
            run_name = os.path.basename(self.work_dir) or os.path.basename(os.path.dirname(self.work_dir)) or "default-run"
        except Exception:
            run_name = "default-run"

        wandb.init(
            project="Open-Vocabulary-Affordance",
            name=run_name,
            config=cleaned_config_dict # クリーンアップ後の辞書を渡す
        )

        # --- 残りの初期化 ---
        self.model = running["model"]
        wandb.watch(self.model, log="all", log_freq=100)
        # ... (以下、変更なし) ...
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
        self.model.train()
        torch.set_grad_enabled(True)

        num_batches = len(self.train_loader)
        start = time()
        self.logger.cprint("Epoch(%d) begin training........" % self.epoch)

        # --- Assign tqdm loop to pbar ---
        pbar = tqdm(self.train_loader, total=len(self.train_loader), smoothing=0.9)
        for data, label in pbar: # Use pbar here
            # --- Initialize batch variables ---
            afford_pred, afford_pred_flat, label_flat, loss = None, None, None, None
            current_loss_item = 0.0 # Initialize loss item

            data, label = data.float().cuda(non_blocking=True), label.float().cuda(non_blocking=True) # Use non_blocking
            label = label.long().contiguous()
            batch_size = data.size()[0]
            num_point_data = data.size()[2]

            self.optimizer.zero_grad()

            with autocast('cuda'):
                afford_pred = self.model(data, self.train_affordance)
                afford_pred = afford_pred.contiguous()

                B_data = data.size(0)

                # --- Reshaping and Permuting (Keep original logic) ---
                if afford_pred.dim() == 2:
                    # (Your existing 2D -> 3D reshape logic here)
                    M_flat, C_out_flat = afford_pred.shape
                    N_data_shape = data.shape[2] # Corrected variable name if needed
                    M = M_flat // B_data
                    if M_flat % B_data != 0:
                         if B_data == 1 and M_flat > C_out_flat: M = M_flat
                         else: raise ValueError(f"Cannot reshape afford_pred [B*M, C] (shape {afford_pred.shape}) with B={B_data}")
                    # print(f"[DEBUG] Reshaping 2D afford_pred {afford_pred.shape} -> 3D [{B_data}, {M}, {C_out_flat}]")
                    afford_pred = afford_pred.view(B_data, M, C_out_flat)

                if afford_pred.dim() == 3:
                    # [B, C, N] -> [B, N, C]
                    afford_pred = afford_pred.permute(0, 2, 1).contiguous()
                    B_out, M, C_out = afford_pred.shape
                else:
                    raise ValueError(f"Unexpected afford_pred dim after normalization: {afford_pred.dim()}")

                # --- Label Squeezing (Keep original logic) ---
                if label.ndim == 3:
                    if label.shape[1] == 1: label = label.squeeze(1)
                    elif label.shape[2] == 1: label = label.squeeze(2)

                if label.ndim == 2: N_label_shape = label.shape[1]
                elif label.ndim == 1: N_label_shape = label.shape[0]
                else: raise ValueError(f"label shape不正: {label.shape}")

                # --- Up/Down Sampling (Keep original logic) ---
                if M != N_label_shape:
                    if N_label_shape > M:
                        # This might be incorrect - typically predictions are upsampled, not labels downsampled
                        # Check if labels should be used to index points OR predictions upsampled
                        print(f"[WARN] afford_pred points ({M}) != label points ({N_label_shape}). Truncating afford_pred.")
                        afford_pred = afford_pred[:, :N_label_shape, :].contiguous()
                    else:
                        idx = torch.linspace(0, M - 1, steps=N_label_shape).long().to(afford_pred.device)
                        afford_pred = torch.index_select(afford_pred, 1, idx).contiguous()
                        print(f"[INFO] Downsampled afford_pred from {M} -> {N_label_shape} points.")

                B_out_up, N_out_up, C_out_up = afford_pred.shape

                # Flatten for loss
                afford_pred_flat = afford_pred.contiguous().view(B_out_up * N_out_up, C_out_up)
                label_flat = label.contiguous().view(-1)

                if label_flat.dtype != torch.long: label_flat = label_flat.long()

                # Truncate if shapes mismatch (Keep original logic)
                if afford_pred_flat.shape[0] != label_flat.shape[0]:
                    min_flat = min(afford_pred_flat.shape[0], label_flat.shape[0])
                    print(f"[WARN] afford_pred_flat/label_flat size mismatch ({afford_pred_flat.shape[0]}, {label_flat.shape[0]}), truncating to {min_flat}")
                    afford_pred_flat = afford_pred_flat[:min_flat, :]
                    label_flat = label_flat[:min_flat]

                loss = self.loss(afford_pred_flat, label_flat)

            # === Mixed Precision ===
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            count += batch_size * num_point_data
            # --- Get loss item before deleting loss tensor ---
            current_loss_item = loss.item()
            train_loss += current_loss_item

            # --- Explicitly delete tensors ---
            try:
                del data, label, afford_pred, afford_pred_flat, label_flat, loss
            except Exception as e:
                self.logger.cprint(f"[WARN] Failed to del tensors in train loop: {e}")

            # --- Update tqdm description ---
            pbar.set_description(f"Epoch {self.epoch}, Loss: {current_loss_item:.6f}")

        # --- Calculate avg_loss after loop ---
        avg_loss = train_loss / num_batches if num_batches > 0 else 0.0

        self.scheduler.step()
        if self.bn_momentum != None:
            self.model.apply(lambda x: self.bn_momentum(x, self.epoch)) # Assuming bn_momentum is a callable function

        epoch_time = time() - start
        outstr = 'Train(%d), loss: %.6f, time: %d s' % (
            self.epoch, avg_loss, epoch_time // 1) # Use avg_loss
        self.writer.add_scalar('train/Loss', avg_loss, self.epoch) # Add prefix 'train/'
        # --- Log avg_loss to wandb ---
        wandb.log({
            "epoch": self.epoch,
            "train/loss": avg_loss,
            "train/epoch_time_sec": epoch_time // 1,
            "train/learning_rate": self.optimizer.param_groups[0]['lr']
        })
        self.logger.cprint(outstr)
        self.epoch += 1
        torch.cuda.empty_cache() # Clear cache after epoch

    def val(self):
        self.logger.cprint('Epoch(%d) begin validating......' % (self.epoch - 1)) # Use self.epoch - 1
        self.model.eval()
        mIoU = 0.0 # Initialize mIoU
        try: # Add try block for evaluation
            with torch.no_grad():
                # Make sure 'evaluation' is correctly imported and defined
                mIoU = evaluation(self.logger, self.model,
                                  self.val_loader, self.val_affordance)
        except Exception as e:
             self.logger.cprint(f"[ERROR] Evaluation failed: {e}")
             # Optionally log error to wandb
             # wandb.log({"eval_error": str(e)})

        # Log metrics to wandb
        wandb.log({
            "epoch": self.epoch - 1, # Log against the epoch that just finished training
            "val/mIoU": mIoU
        })

        # Save best model
        if mIoU >= self.best_val_mIoU:
            self.best_val_mIoU = mIoU
            self.logger.cprint('Saving best model......')
            self.logger.cprint('Best mIoU: %f' % self.best_val_mIoU)
            best_model_path = opj(self.work_dir, 'best_model.t7') # Define path
            torch.save(self.model.state_dict(), best_model_path)

            # --- Correct wandb.save call (inside if block) ---
            try:
                 # Use glob_str and ensure base_path exists
                 wandb.save(glob_str=best_model_path, base_path=self.work_dir, policy="live")
            except Exception as e:
                 self.logger.cprint(f"[WARN] Failed to save best model to wandb: {e}")


        # Save current model
        current_model_path = opj(self.work_dir, 'current_model.t7')
        torch.save(self.model.state_dict(), current_model_path)
        # Optionally save current model to wandb too
        # try:
        #     wandb.save(glob_str=current_model_path, base_path=self.work_dir, policy="live")
        # except Exception as e:
        #      self.logger.cprint(f"[WARN] Failed to save current model to wandb: {e}")

        # --- Add cuda empty_cache ---
        torch.cuda.empty_cache()


    def run(self):
        EPOCH = self.cfg.training_cfg.epoch
        workflow = self.cfg.training_cfg.workflow
        try:
            while self.epoch < EPOCH:
                for key, running_epoch in workflow.items():
                    epoch_runner = getattr(self, key)
                    for e in range(running_epoch):
                        # Add check if loader exists (e.g., if only validation is run)
                        if key == "train" and not self.train_loader:
                            self.logger.cprint("Skipping train epoch: train_loader not found.")
                            continue
                        if key == "val" and not self.val_loader:
                             self.logger.cprint("Skipping val epoch: val_loader not found.")
                             continue
                        epoch_runner()
        except KeyboardInterrupt: # Handle Ctrl+C gracefully
             self.logger.cprint("Training interrupted by user.")
        except Exception as e:
            self.logger.cprint(f"[ERROR] Training failed: {e}")
            import traceback
            self.logger.cprint(traceback.format_exc()) # Log full traceback
            # Log error to wandb
            wandb.log({"error": str(e)})
        finally:
            self.logger.cprint("Training finished. Finishing wandb run.")
            wandb.finish() # Ensure wandb finishes even on error