import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_util import PointNetSetAbstraction, PointNetSetAbstractionMsg, PointNetFeaturePropagation

class SceneFun3DPN2(nn.Module):
    def __init__(self, num_class=20, use_msg=True):
        super().__init__()
        self.num_class = num_class

        in_channel = 0  # XYZ のみ
        # Set Abstraction 層
        if use_msg:
            self.sa1 = PointNetSetAbstractionMsg(
                npoint=512,
                radius_list=[0.1, 0.2, 0.4],
                nsample_list=[32, 64, 128],
                in_channel=in_channel,
                mlp_list=[[32,32,64],[64,64,128],[64,96,128]]
            )
            sa1_out = 64+128+128  # MSG concat 出力
        else:
            self.sa1 = PointNetSetAbstraction(
                npoint=512,
                radius=0.2,
                nsample=64,
                in_channel=in_channel,
                mlp=[64,128,128],
                group_all=False
            )
            sa1_out = 128

        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=sa1_out,
            mlp=[256,256,512],
            group_all=False
        )

        # Feature Propagation
        self.fp2 = PointNetFeaturePropagation(in_channel=512+sa1_out, mlp=[256,256])
        self.fp1 = PointNetFeaturePropagation(in_channel=256+3, mlp=[256,128,128])

        # 最終 MLP (affordance prediction)
        self.final_mlp = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, self.num_class, 1)
        )

    def forward(self, xyz, points=None):
        """
        xyz: [B, N, 3]
        points: [B, N, C] or None
        """
        # SA 層
        l1_xyz, l1_points = self.sa1(xyz, points)   # [B,3,512], [B,C1,512]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B,3,128], [B,C2,128]

        # FP 層
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # [B,C,512]
        l0_points = self.fp1(xyz.permute(0,2,1), l1_xyz, None, l1_points)  # [B,C, N]

        # 最終 MLP
        afford_pred = self.final_mlp(l0_points)  # [B, num_class, N]
        return afford_pred
