import os
from os.path import join as opj
import numpy as np
from torch.utils.data import Dataset
import pickle as pkl
import h5py as h5


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m


class AffordNetDataset(Dataset):
    def __init__(self, data_dir, split, partial=False):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.partial = partial

        self.load_data()
        self.affordances = self.all_data[0]["affordance"]
        return

    def load_data(self):
        self.all_data = []

        if self.partial:
            with open(opj(self.data_dir, 'partial_view_%s_data.pkl' % self.split), 'rb') as f:
                temp_data = pkl.load(f)
        else:
            # train/valファイルリストを読み込み
            file_list_path = opj(self.data_dir, '%s_files.txt' % self.split)
            with open(file_list_path, 'r') as f:
                file_list = [line.strip() for line in f.readlines()]
            
            temp_data = []
            for filename in file_list:
                file_path = opj(self.data_dir, filename)
                with h5.File(file_path, 'r') as f:
                    # HDF5ファイルからデータを読み込み
                    data = f['data'][:]  # 形状: (100, 4096, 6)
                    label = f['label'][:]  # 形状: (100, 4096)
                    
                    # ファイル名からshape_idを抽出
                    shape_id = filename.replace('.h5', '')
                    
                    # 各オブジェクトを個別のデータとして追加
                    for obj_idx in range(data.shape[0]):
                        info = {
                            'shape_id': f"{shape_id}_{obj_idx}",
                            'semantic class': 'unknown',  # デフォルト値
                            'affordance': ['grasp'],  # デフォルト値
                            'full_shape': {
                                'coordinate': data[obj_idx],  # 形状: (4096, 6)
                                'label': label[obj_idx]  # 形状: (4096,)
                            }
                        }
                        temp_data.append(info)
        for _, info in enumerate(temp_data):
            if self.partial:
                partial_info = info["partial"]
                for view, data_info in partial_info.items():
                    temp_info = {}
                    temp_info["shape_id"] = info["shape_id"]
                    temp_info["semantic class"] = info["semantic class"]
                    temp_info["affordance"] = info["affordance"]
                    temp_info["view_id"] = view
                    temp_info["data_info"] = data_info
                    self.all_data.append(temp_info)
            else:
                temp_info = {}
                temp_info["shape_id"] = info["shape_id"]
                temp_info["semantic class"] = info["semantic class"]
                temp_info["affordance"] = info["affordance"]
                temp_info["data_info"] = info["full_shape"]
                self.all_data.append(temp_info)

    def __getitem__(self, index):

        data_dict = self.all_data[index]
        modelid = data_dict["shape_id"]
        modelcat = data_dict["semantic class"]

        data_info = data_dict["data_info"]
        model_data = data_info["coordinate"].astype(np.float32)  # 形状: (4096, 6)
        labels = data_info["label"]  # 形状: (4096,)
        
        # ラベルを処理：255をignore label(-100)に、0-8を0-18にマッピング
        processed_labels = labels.copy()
        processed_labels[processed_labels == 255] = -100  # ignore label
        processed_labels[processed_labels > 8] = 0  # 範囲外の値を0にマッピング
        
        # ラベルを2次元配列に変換
        temp = processed_labels.astype(np.float32).reshape(-1, 1)  # 形状: (4096, 1)
        
        # 座標データとラベルを結合
        model_data = np.concatenate((model_data, temp), axis=1)  # 形状: (4096, 7)

        datas = model_data[:, :3]  # xyz座標: (4096, 3)
        targets = model_data[:, 3]  # ラベル: (4096,) - 1次元配列

        datas, _, _ = pc_normalize(datas)

        # trainerが期待する形式に合わせて2つの値のみを返す
        # データを [N, C] から [B, C, N] に変換（PointNet++の期待形式）
        datas = datas.T.reshape(1, 3, -1)  # (1, 3, 4096)
        targets = targets.reshape(1, -1)  # (1, 4096) - ラベルは1次元
        
        return datas, targets

    def __len__(self):
        return len(self.all_data)