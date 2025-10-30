# (utils.py の末尾に追加)

import torch.utils.data as data
import h5py

# ----------------------------------------------------
# 1. SceneFun3D データセットクラス
# ----------------------------------------------------
class SceneFun3DDataset(data.Dataset):
    """
    SceneFun3Dデータセット用のカスタムDatasetクラス (セマンティックセグメンテーション用)
    .h5 ファイルを読み込むことを想定しています。
    """
    def __init__(self, data_root, split='train', num_points=4096, use_color=False):
        """
        Args:
            data_root (str): データセットのルートディレクトリ
            split (str): 'train' または 'val'
            num_points (int): 1つのサンプル（ブロック）からサンプリングする点の数
            use_color (bool): 色情報を特徴として使用するかどうか
        """
        self.data_root = data_root
        self.split = split
        self.num_points = num_points
        self.use_color = use_color

        # .h5 ファイルのリストを取得
        file_list_path = os.path.join(self.data_root, f'{split}_files.txt')
        if not os.path.exists(file_list_path):
            raise FileNotFoundError(f"ファイルリストが見つかりません: {file_list_path}")
            
        with open(file_list_path, 'r') as f:
            self.file_list = [line.strip() for line in f]
            
        # データをメモリにロード
        # (メモリが足りない場合は、__getitem__ 内で都度ロードする方式に変更します)
        self.all_data = []
        self.all_labels = []
        for h5_name in self.file_list:
            # h5_name が data_root からの相対パスでない場合、結合する
            file_path = os.path.join(self.data_root, h5_name)
            if not os.path.exists(file_path):
                 # .txt ファイルがファイル名のみを含んでいる場合
                 file_path = os.path.join(self.data_root, h5_name) 
                 # (注: SceneFun3D の .txt がどういう形式かによります)
                 # ここでは .txt に書かれているのが .h5 のファイル名と仮定します

            f = h5py.File(file_path, 'r')
            data = f['data'][:]  # (B, N, C)
            label = f['label'][:] # (B, N)
            f.close()
            self.all_data.append(data)
            self.all_labels.append(label)
            
        # (B, N, C) のリストを (Total_B, N, C) の単一配列に結合
        self.all_data = np.concatenate(self.all_data, axis=0)
        self.all_labels = np.concatenate(self.all_labels, axis=0)

        print(f"SceneFun3D {split} データセット読み込み完了。")
        print(f"データ形状: {self.all_data.shape}, ラベル形状: {self.all_labels.shape}")


    def __len__(self):
        """データセットの総サンプル数（総ブロック数）を返す"""
        return self.all_data.shape[0]

    def __getitem__(self, index):
        """
        指定されたインデックスのデータを取得
        """
        # (N, C)
        point_cloud_data = self.all_data[index]
        # (N,)
        label_data = self.all_labels[index]

        # 1. 点のサンプリング (num_points に合わせる)
        num_points_in_block = point_cloud_data.shape[0]
        
        if num_points_in_block >= self.num_points:
            indices = np.random.choice(num_points_in_block, self.num_points, replace=False)
        else:
            indices = np.random.choice(num_points_in_block, self.num_points, replace=True)
            
        sampled_points = point_cloud_data[indices, :]
        sampled_labels = label_data[indices]

        # 2. 特徴量の選択 (XYZ または XYZ + Color)
        # SceneFun3D の .h5 は通常 (XYZ, RGB, Normals) の9次元です
        if self.use_color:
            features = sampled_points[:, :6] # XYZRGB
            # (もし正規化座標も使うなら 0:3 と 6:9 を使うなど、調整が必要です)
        else:
            features = sampled_points[:, :3] # XYZ のみ
            
        # (オプション: データ拡張をここに追加)
        # ...

        # 3. PyTorch Tensor に変換
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(sampled_labels, dtype=torch.long)
        
        # (B, C, N) 形式を期待するモデルのため転置
        features = features.transpose(0, 1) # (C, num_points)

        return features, labels


# ----------------------------------------------------
# 2. build_dataset 関数
# ----------------------------------------------------
def build_dataset(cfg):
    """
    cfg に基づいてデータセットを構築する
    """
    dataset_type = cfg.dataset.get('type', 'SceneFun3DDataset')

    if dataset_type == 'SceneFun3DDataset':
        train_cfg = cfg.dataset.train
        train_dataset = SceneFun3DDataset(
            data_root=train_cfg.data_root,
            split='train',
            num_points=train_cfg.get('num_points', 4096),
            use_color=train_cfg.get('use_color', False)
        )
        
        val_cfg = cfg.dataset.val
        val_dataset = SceneFun3DDataset(
            data_root=val_cfg.data_root,
            split='val',
            num_points=val_cfg.get('num_points', 4096),
            use_color=val_cfg.get('use_color', False)
        )
        
        dataset_dict = {
            'train': train_dataset,
            'val': val_dataset
        }
        return dataset_dict
        
    else:
        raise NotImplementedError(f"データセットタイプ '{dataset_type}' はサポートされていません。")

# ----------------------------------------------------
# 3. build_loader 関数 (必須)
# ----------------------------------------------------
def build_loader(cfg, dataset_dict):
    """
    cfg と dataset_dict からデータローダーを構築する
    """
    train_loader = data.DataLoader(
        dataset_dict['train'],
        batch_size=cfg.data_loader.train.batch_size,
        shuffle=cfg.data_loader.train.shuffle,
        num_workers=cfg.data_loader.train.num_workers,
        pin_memory=True # GPU訓練時に高速化
    )
    
    val_loader = data.DataLoader(
        dataset_dict['val'],
        batch_size=cfg.data_loader.val.batch_size,
        shuffle=cfg.data_loader.val.shuffle,
        num_workers=cfg.data_loader.val.num_workers,
        pin_memory=True
    )
    
    loader_dict = {
        'train': train_loader,
        'val': val_loader
    }
    return loader_dict

# ----------------------------------------------------
# 4. build_loss 関数 (必須)
# ----------------------------------------------------
def build_loss(cfg):
    """
    cfg から損失関数を構築する
    """
    loss_type = cfg.loss.get('type', 'CrossEntropyLoss')
    
    if loss_type == 'CrossEntropyLoss':
        # (例: ラベル 0 を無視する場合)
        # ignore_index = cfg.loss.get('ignore_index', 0) 
        # return nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        # シンプルなクロスエントロピー
        return nn.CrossEntropyLoss()
    
    # (他の損失関数 ...例: 'NLLLoss')
    # elif loss_type == 'NLLLoss':
    #     return nn.NLLLoss()
        
    else:
        raise NotImplementedError(f"損失関数タイプ '{loss_type}' はサポートされていません。")

# ----------------------------------------------------
# 5. build_optimizer 関数 (必須)
# ----------------------------------------------------
def build_optimizer(cfg, model):
    """
    cfg と model からオプティマイザとスケジューラを構築する
    """
    optim_type = cfg.optimizer.get('type', 'Adam')
    
    optimizer = None
    if optim_type == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.lr,
            # (必要なら weight_decay などを cfg から取得)
        )
    elif optim_type == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.get('momentum', 0.9),
            weight_decay=cfg.optimizer.get('weight_decay', 1e-4)
        )
    else:
        raise NotImplementedError(f"オプティマイザタイプ '{optim_type}' はサポートされていません。")

    # スケジューラの構築
    scheduler = None
    if cfg.get('lr_scheduler') is not None:
        scheduler_type = cfg.lr_scheduler.get('type')
        
        if scheduler_type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=cfg.lr_scheduler.step_size,
                gamma=cfg.lr_scheduler.gamma
            )
        elif scheduler_type == 'PN2_Scheduler': # utils.py にあるカスタムスケジューラ
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=PN2_Scheduler(
                    init_lr=cfg.optimizer.lr,
                    step=cfg.lr_scheduler.step,
                    decay_rate=cfg.lr_scheduler.decay_rate,
                    min_lr=cfg.lr_scheduler.min_lr
                )
            )
        # (他のスケジューラも追加可能)

    return dict(
        optimizer=optimizer,
        scheduler=scheduler
    )

# ----------------------------------------------------
# 6. build_model 関数 (必須)
# ----------------------------------------------------
def build_model(cfg):
    """
    cfg からモデルを構築する
    (注: この関数はモデルの定義に依存します)
    """
    model_type = cfg.model.get('type')
    
    # (重要)
    # ここに、使用するモデル（PointNet, PointNet++, DGCNN など）を
    # インポートしてインスタンス化するコードが必要です。
    
    # (例: 別のファイル `models.py` にモデルが定義されている場合)
    # from . import models
    #
    # if model_type == 'PointNet2Seg':
    #     model = models.PointNet2Seg(
    #         num_classes=cfg.model.num_classes,
    #         input_channels=cfg.model.input_channels
    #     )
    # elif model_type == 'DGCNN_seg':
    #     model = models.DGCNN_seg(
    #         num_classes=cfg.model.num_classes,
    #         input_channels=cfg.model.input_channels
    #     )
    # else:
    #     raise NotImplementedError(f"モデルタイプ '{model_type}' はサポートされていません。")
    #
    # return model

    # --- (仮の実装) ---
    # モデル定義がないため、ここではエラーを出すか、
    # 非常に単純なダミーモデルを返します。
    # ☆ あとで必ず実際のモデル定義に置き換えてください ☆
    print(f"警告: 'build_model' が仮実装です。'{model_type}' をロードしようとしました。")
    print("実際のモデルクラス (PointNet など) をインポートして構築する必要があります。")
    
    # 仮のダミーモデル (入力 (B,C,N), 出力 (B, NumClasses, N))
    # (注: これでは訓練できません！)
    class DummyModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.in_channels = cfg.model.get('input_channels', 6)
            self.num_classes = cfg.model.get('num_classes', 13)
            # (B, C, N) -> (B, 64, N)
            self.conv1 = nn.Conv1d(self.in_channels, 64, 1)
            # (B, 64, N) -> (B, NumClasses, N)
            self.conv2 = nn.Conv1d(64, self.num_classes, 1)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.conv2(x)
            return x
            
    return DummyModel(cfg)


# ----------------------------------------------------
# 7. Trainer クラス (必須)
# ----------------------------------------------------
# (前回の回答で示した `Trainer` クラスの例をここに追加します)

class Trainer:
    def __init__(self, cfg, training):
        self.cfg = cfg
        self.model = training['model']
        self.logger = training['logger']
        self.loader_dict = training['loader_dict']
        self.train_loader = self.loader_dict.get('train')
        self.val_loader = self.loader_dict.get('val')
        self.loss = training['loss']
        self.optimizer = training['optim_dict']['optimizer']
        self.scheduler = training['optim_dict']['scheduler']
        
        self.work_dir = cfg.work_dir
        self.max_epoch = cfg.training_cfg.max_epoch
        self.log_freq = cfg.training_cfg.get('log_freq', 10)
        self.val_freq = cfg.training_cfg.get('val_freq', 1)

        # ベスト性能を記録
        self.best_val_mIoU = 0.0 # (仮に mIoU を基準にします)
        
        # (オプション: BNモーメンタムスケジューラ)
        self.bn_momentum_scheduler = None
        if cfg.get('bn_momentum_scheduler') is not None:
            self.bn_momentum_scheduler = PN2_BNMomentum(
                origin_m=cfg.bn_momentum_scheduler.origin_m,
                m_decay=cfg.bn_momentum_scheduler.m_decay,
                step=cfg.bn_momentum_scheduler.step
            )

    def run(self):
        """ メインの訓練ループ """
        for epoch in range(1, self.max_epoch + 1):
            
            # (オプション: BNモーメンタム更新)
            if self.bn_momentum_scheduler is not None:
                self.model.apply(lambda m: self.bn_momentum_scheduler(m, epoch))
            
            # --- 1. 訓練 ---
            self.run_epoch(epoch, 'train')
            
            # --- 2. 学習率の更新 ---
            if self.scheduler is not None:
                self.scheduler.step()

            # --- 3. 検証 ---
            if epoch % self.val_freq == 0:
                self.logger.cprint(f"--- Epoch {epoch} Validation ---")
                
                current_val_metric = self.run_epoch(epoch, 'val') # (例: mIoU)
                
                self.logger.cprint(f"Validation Metric (mIoU/Accuracy): {current_val_metric:.4f}")
                
                # --- 4. ベストモデルの保存 ---
                if current_val_metric > self.best_val_mIoU:
                    self.best_val_mIoU = current_val_metric
                    self.logger.cprint(f"*** New Best Metric found: {self.best_val_mIoU:.4f} ***")
                    
                    save_path = opj(self.work_dir, 'best_model.t7')
                    try:
                        torch.save(self.model.state_dict(), save_path)
                        self.logger.cprint(f"Best model saved to: {save_path}")
                    except Exception as e:
                        self.logger.cprint(f"Error saving best model: {e}")
            
            # (オプション: 定期的にチェックポイントを保存)
            # if epoch % cfg.training_cfg.save_freq == 0:
            #    ... (save 'checkpoint.pth') ...

        self.logger.cprint("Training finished.")


    def run_epoch(self, epoch, split):
        """ 1エポックの訓練または検証を実行 """
        
        if split == 'train':
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.val_loader
            # (注: ここでは mIoU を計算する代わりに、仮のメトリクス（精度）を計算します)
            # (正確な mIoU 計算ロジックが必要です)
            total_metric = 0.0
            num_batches = 0

        total_loss = 0.0
        
        for i, (data, labels) in enumerate(loader):
            data, labels = data.cuda(), labels.cuda() # (B, C, N), (B, N)

            if split == 'train':
                # --- 訓練 ---
                self.optimizer.zero_grad()
                
                preds = self.model(data) # (B, NumClasses, N)
                loss = self.loss(preds, labels)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if (i + 1) % self.log_freq == 0:
                    self.logger.cprint(f"Epoch {epoch} [TRAIN] Batch {i+1}/{len(loader)}: Loss={loss.item():.4f}")

            else:
                # --- 検証 ---
                with torch.no_grad():
                    preds = self.model(data) # (B, NumClasses, N)
                    loss = self.loss(preds, labels)
                    
                    total_loss += loss.item()

                    # メトリクス計算 (仮: 精度)
                    # ☆ ここを SceneFun3D の mIoU 計算に置き換えてください ☆
                    pred_labels = torch.argmax(preds, dim=1) # (B, N)
                    correct = (pred_labels == labels).sum().item()
                    batch_metric = correct / (labels.shape[0] * labels.shape[1])
                    
                    total_metric += batch_metric
                    num_batches += 1

        if split == 'val':
            avg_metric = total_metric / num_batches
            avg_loss = total_loss / num_batches
            self.logger.cprint(f"Epoch {epoch} [VAL] Avg Loss: {avg_loss:.4f}, Avg Metric: {avg_metric:.4f}")
            return avg_metric # 検証メトリクスを返す
        else:
            avg_loss = total_loss / len(loader)
            self.logger.cprint(f"Epoch {epoch} [TRAIN] Avg Loss: {avg_loss:.4f}")
            return avg_loss # 訓練損失を返す