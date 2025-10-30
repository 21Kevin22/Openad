import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EstimationLoss(nn.Module):
    def __init__(self, cfg):
        super(EstimationLoss, self).__init__()

        # builder.py で設定された ignore_label を cfg から取得します
        # (デフォルト値は -100 など、通常使われない値にしておきます)
        ignore_idx = cfg.training_cfg.get('ignore_label', -100) 
        
        # (デバッグ用) 損失関数がどの ignore_index で初期化されたか確認
        print(f"[INFO] EstimationLoss: Initializing CrossEntropyLoss with ignore_index={ignore_idx}")
        
        # nn.CrossEntropyLoss の初期化時に ignore_index を渡します
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_idx)
        self.weights = torch.from_numpy(np.load(cfg.training_cfg.weights_dir)).cuda().float()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target, weight=self.weights)
        return total_loss
