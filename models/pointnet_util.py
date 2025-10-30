import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time import time

# pointnet2_opsの代わりにPyTorchの標準機能を使用
# from .pointnet2_ops import pointnet2_utils as pointnet2_util

# ====================== 共通ユーティリティ ======================

def square_distance(src, dst):
    """
    (B, N, C) と (B, M, C) の間の二乗距離を計算 -> (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    points: (B, N, C)
    idx: (B, S, K) または (B, S)
    return: (B, S, K, C) または (B, S, C)
    """
    device = points.device
    B, N, C = points.shape
    idx = idx.long().to(device)
    
    # idx が範囲外にならないようにクリップする
    # (query_ball_point が N を返す場合などへの対処)
    idx = torch.clamp(idx, 0, N - 1)

    if idx.dim() == 2:
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, C)
        return torch.gather(points, 1, idx_expanded)
    elif idx.dim() == 3:
        B2, S, K = idx.shape
        # (B, 1, N, C) -> (B, S, N, C)
        points_expanded = points.unsqueeze(1).expand(-1, S, -1, -1)
        # (B, S, K, 1) -> (B, S, K, C)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, C)
        # (B, S, N, C) から (B, S, K, C) を収集
        return torch.gather(points_expanded, 2, idx_expanded)
    else:
        raise ValueError(f"idx must be 2D or 3D, got {idx.shape}")

def farthest_point_sample(xyz, npoint):
    """
    Farthest point sampling
    xyz: (B, N, 3)
    npoint: S
    return: (B, S) (インデックス)
    """
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((B, N), 1e10, device=xyz.device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query
    xyz: (B, N, 3)
    new_xyz: (B, S, 3)
    radius: float
    nsample: K
    return: (B, S, K) (インデックス)
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    nsample_eff = min(nsample, N) # サンプル数 K は N を超えられない
    
    group_idx = torch.arange(N, device=xyz.device).view(1,1,N).expand(B,S,N)
    
    # ✅ 修正点 1 (UserWarning対処): 
    # expand されたテンソルへのインプレース代入を避けるため clone()
    group_idx = group_idx.clone()
    
    sqrdists = square_distance(new_xyz, xyz) # (B, S, N)
    
    # 半径外の点のインデックスを N (無効な値) に設定
    group_idx[sqrdists > radius**2] = N
    
    # ソートして K 個取得 (半径内の点が手前に来る)
    group_idx = torch.sort(group_idx, dim=-1)[0][:,:,:nsample_eff]
    
    # 半径内に K 個未満の点しかない場合、最も近い点 (group_idx[0]) でパディング
    group_first = group_idx[:,:,0:1].expand(-1,-1,nsample_eff)
    group_idx = torch.where(group_idx==N, group_first, group_idx)

    # K が N より大きい場合 (nsample > nsample_eff)、最後の点でパディング
    if nsample > nsample_eff:
        last_idx = group_idx[:,:,-1:].expand(-1,-1, nsample-nsample_eff)
        group_idx = torch.cat([group_idx, last_idx], dim=-1)
        
    return group_idx
    if label.ndim == 3:
            # (B, 1, N) -> (B, N) を想定
            if label.shape[1] == 1:
                label = label.squeeze(1)
            # (B, N, 1) -> (B, N) を想定
            elif label.shape[2] == 1:
                label = label.squeeze(2)

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Sampling and grouping
    npoint: S
    radius: float
    nsample: K
    xyz: (B, N, 3)
    points: (B, N, D)
    return:
        new_xyz: (B, S, 3)
        new_points: (B, S, K, 3+D)
    """
    fps_idx = farthest_point_sample(xyz, npoint) # (B, S)
    new_xyz = index_points(xyz, fps_idx) # (B, S, 3)
    
    idx = query_ball_point(radius, nsample, xyz, new_xyz) # (B, S, K)
    
    grouped_xyz = index_points(xyz, idx) # (B, S, K, 3)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, 3) # 相対座標

    if points is not None:
        grouped_points = index_points(points, idx) # (B, S, K, D)
        # 座標(3)と特徴(D)を結合
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # (B, S, K, 3+D)
    else:
        new_points = grouped_xyz_norm
        
    return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Group all points
    xyz: (B, N, 3)
    points: (B, N, D)
    return:
        new_xyz: (B, 1, 3) (原点)
        new_points: (B, 1, N, 3+D)
    """
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B,1,C,device=xyz.device) # (B, 1, 3)
    
    grouped_xyz = xyz.view(B,1,N,C) # (B, 1, N, 3)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B,1,N,-1)], dim=-1) # (B, 1, N, 3+D)
    else:
        new_points = grouped_xyz
        
    return new_xyz, new_points

def _safe_conv_apply(conv, bn, x, grouped_xyz=None, layer_name=None):
    # (B, C_in, K, S) -> (B, C_out, K, S)
    return F.relu(bn(conv(x)))

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint, self.radius, self.nsample, self.group_all = npoint, radius, nsample, group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        # PointNetSetAbstraction では in_channel は 3+D (XYZ+features) を受け取ると仮定
        last_channel = in_channel 
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel


    def forward(self, xyz, points):
        """
        xyz: (B, 3, N)
        points: (B, D, N)
        return:
            new_xyz: (B, 3, S)
            new_points: (B, C_out, S)
        """
        
        # (B, C, N) -> (B, N, C) 形式に変換
        if xyz.shape[1] == 3: xyz = xyz.permute(0,2,1).contiguous()
        if points is not None:
            points = points.permute(0,2,1).contiguous()

        # サンプリング＆グループ化 (new_points は 3+D チャンネルを持つ)
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: (B, S, 3)
        # new_points: (B, S, K, 3+D)

        # (B, S, K, C_in) -> (B, C_in, K, S)
        new_points = new_points.permute(0,3,2,1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = _safe_conv_apply(conv, bn, new_points)
        
        # Max pooling (K次元に対して)
        new_points = torch.max(new_points, 2)[0] # (B, C_out, S)
        
        return new_xyz.permute(0,2,1), new_points # (B, 3, S), (B, C_out, S)


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super().__init__()
        self.npoint, self.radius_list, self.nsample_list = npoint, radius_list, nsample_list
        self.conv_blocks, self.bn_blocks = nn.ModuleList(), nn.ModuleList()
        
        for mlp_spec in mlp_list:
            convs, bns = nn.ModuleList(), nn.ModuleList()
            
            # ✅ 修正点 2 (RuntimeError対処):
            # PointNetSetAbstractionMsg では in_channel は D (features) を受け取ると仮定
            # (forward で 3+D になるため)
            last_channel = in_channel + 3
            for out_channel in mlp_spec:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        xyz: (B, 3, N)
        points: (B, D, N)
        return:
            new_xyz: (B, 3, S)
            new_points: (B, C_total, S)
        """
        # (B, C, N) -> (B, N, C) 形式に変換
        if xyz.shape[1]==3: xyz = xyz.permute(0,2,1).contiguous()
        if points is not None:
            points = points.permute(0,2,1).contiguous()
            
        B, N, C = xyz.shape
        S = self.npoint
        
        # Farthest point sampling
        new_xyz = index_points(xyz, farthest_point_sample(xyz,S)) # (B, S, 3)
        
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius,K,xyz,new_xyz) # (B, S, K)
            
            grouped_xyz = index_points(xyz, group_idx) # (B, S, K, 3)
            grouped_xyz -= new_xyz.view(B,S,1,C) # 相対座標
            
            if points is not None:
                grouped_points = index_points(points, group_idx) # (B, S, K, D)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1) # (B, S, K, 3+D)
            else:
                grouped_points = grouped_xyz # (B, S, K, 3)

            # (B, S, K, C_in) -> (B, C_in, K, S)
            # C_in は 3+D となり、__init__ で設定した Conv2d(3+D, ...) と一致する
            grouped_points = grouped_points.permute(0,3,2,1)
            
            for j, (conv,bn) in enumerate(zip(self.conv_blocks[i], self.bn_blocks[i])):
                grouped_points = _safe_conv_apply(conv, bn, grouped_points, layer_name=f"MSG{i}")
            
            new_points = torch.max(grouped_points, 2)[0] # (B, C_out_i, S)
            new_points_list.append(new_points)
            
        return new_xyz.permute(0,2,1), torch.cat(new_points_list, dim=1)

# ====================== Feature Propagation ======================

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs, self.mlp_bns = nn.ModuleList(), nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def _three_interpolate(self, points, idx, weight):
        """
        3近傍補間の実装
        points: (B, M, C_in) 特徴 (permute済み)
        idx: (B, N, K) 補間に使うM側の点のインデックス
        weight: (B, N, K) 補間の重み
        return: (B, C_in, N) 補間された特徴
        """
        # grouped_points: (B, N, K, C_in)
        grouped_points = index_points(points, idx)
        
        # weight: (B, N, K, 1)
        weight = weight.unsqueeze(-1)
        
        # (B, N, K, C_in) * (B, N, K, 1) -> sum(dim=2) -> (B, N, C_in)
        interpolated = torch.sum(grouped_points * weight, dim=2)
        
        # (B, N, C_in) -> (B, C_in, N)
        return interpolated.permute(0, 2, 1)


    def forward(self, unknown, known, points1, points2):
        """
        unknown: (B, N, 3) 補間先の座標 (例: l2_xyz_perm)
        known: (B, M, 3) 補間元の座標 (例: l3_xyz_perm)
        points1: (B, C1, N) 補間先のスキップ特徴 (例: l2_points)
        points2: (B, C2, M) 補間元の特徴 (例: l3_points)
        return: (B, C_out, N)
        """
        if known is not None:
            B, N, _ = unknown.shape
            _, M, _ = known.shape
            
            # 距離計算
            dist = square_distance(unknown, known)  # (B, N, M)
            
            # k=3 を動的に変更 (M が 3 より小さい場合に対応)
            k = min(3, M)
            
            # 距離が小さい順に k 個のインデックスと値を取得
            dist_recip, idx = torch.topk(dist, k=k, dim=-1, largest=False)  # (B, N, k)
            
            # 重み計算 (逆距離加重)
            dist_recip = 1.0 / (dist_recip + 1e-8)  # (B, N, k)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm # (B, N, k)
            
            # points2 (補間元特徴) の形式を (B, M, C2) に変更
            points2_perm = points2.permute(0, 2, 1).contiguous() # (B, M, C2)
            
            interpolated_points = self._three_interpolate(points2_perm, idx, weight) # (B, C2, N)
        else:
            # group_all=True の場合 (known=None)
            interpolated_points = points2.expand(-1, -1, N) # (B, C2, N)

        # スキップ接続 (points1) と補間された特徴 (interpolated_points) を結合
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1) # (B, C1+C2, N)
        else:
            new_points = interpolated_points # (B, C2, N)
        
        # 正しい MLP (Conv1d) の適用
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
            
        return new_points