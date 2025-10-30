import torch
import torch.nn as nn
import clip
import torch.nn.functional as F
import numpy as np
from .pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

class SceneFun3DPN2(nn.Module):
    def __init__(self, model_info, num_category):
        super().__init__()
        self.device = torch.device('cuda')
        self.num_category = num_category
        self.clip_model = None  # 遅延ロード用

        # PointNet++の層を追加（3チャンネル入力用に修正）
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [
                                             32, 64, 128], 3, [[32, 32, 64],\
                                                [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.4, 0.8], [64, 128], 320, [[128, 128, 256], [128, 196, 256]]) # 修正: sa1 の出力 64+128+128=320 に合わせる
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256]) # 1024 (l3) + 512 (l2)
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128]) # 256 (fp3 out) + 320 (l1)
        
        # 修正の可能性: fp1 の in_channel
        # l0_points (xyz, 3ch) + l1_points (fp2 out, 128ch) = 131
        # 元の 134 は 131 の間違いである可能性が高いと仮定し、131 に修正します。
        # もし入力 xyz が 6 チャンネル (XYZRGB) で、fp1 に xyz (6ch) を渡す場合は 134 (128 + 6) が正しいです。
        self.fp1 = PointNetFeaturePropagation(
            in_channel=131, mlp=[128, 128]) # 134 から 131 に修正

        self.conv1 = nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _load_clip_model(self):
        if self.clip_model is None:
            self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def forward(self, xyz, affordance):
        # 点群データの処理
        xyz = xyz.contiguous()
        
        # DataLoaderでバッチ化された場合の形状処理
        if xyz.ndim == 4:  # [B, 1, C, N] -> [B, C, N]
            xyz = xyz.squeeze(1)
        B, C, N = xyz.shape   # (B, 3, N) と仮定 (C=3)

        l0_xyz = xyz
        # sa1 の in_channel=3 に基づき、l0_points は 3 チャンネル (xyz) と仮定
        l0_points = xyz  
        
        # --- 修正点 1: エンコーダ (SA) レイヤーの呼び出し ---
        # l1_xyz, l1_points などを定義します
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) # l1_points: (B, 320, 512)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # l2_points: (B, 512, 128)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # l3_points: (B, 1024, 1)
        # --- 修正ここまで ---

        l0_xyz_perm = l0_xyz.permute(0, 2, 1).contiguous()
        l1_xyz_perm = l1_xyz.permute(0, 2, 1).contiguous()
        l2_xyz_perm = l2_xyz.permute(0, 2, 1).contiguous()
        l3_xyz_perm = l3_xyz.permute(0, 2, 1).contiguous()

        # Feature Propagation layers (デコーダ)
        # l2_points (B, 512, N_l2) と l3_points (B, 1024, 1) から l2_points (B, 256, N_l2) を生成
        l2_points = self.fp3(l2_xyz_perm, l3_xyz_perm, l2_points, l3_points)
        # l1_points (B, 320, N_l1) と l2_points (B, 256, N_l2) から l1_points (B, 128, N_l1) を生成
        l1_points = self.fp2(l1_xyz_perm, l2_xyz_perm, l1_points, l2_points)
        # l0_xyz (B, 3, N) と l1_points (B, 128, N_l1) から l0_points (B, 128, N) を生成
        # (fp1 の in_channel=131 = 3 + 128)
        l0_points = self.fp1(l0_xyz_perm, l1_xyz_perm, l0_xyz, l1_points)


        l0_points = self.bn1(self.conv1(l0_points))

        self._load_clip_model()
        tokens = clip.tokenize(affordance).to(self.device)
        text_features = self.clip_model.encode_text(tokens).to(self.device).permute(1, 0).float()

        l0_points = l0_points.permute(0, 2, 1).float()
        x = (self.logit_scale * (l0_points @ text_features) /
             (torch.norm(l0_points, dim=2, keepdim=True) @ torch.norm(text_features, dim=0, keepdim=True))).permute(0, 2, 1)

        x = F.log_softmax(x, dim=1)
        return x