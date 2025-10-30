import os
import sys
import numpy as np
import h5py
from plyfile import PlyData # Keep PlyData for saving if needed, but DataParser uses open3d
import tqdm # Use import tqdm
import json # Added for JSON parsing
# data_parser のインポートは try-except 内で行う
# from .data_parser import DataParser # スクリプトとして直接実行する場合は . なし


try:
    # BASE_DIR をスクリプトのディレクトリに設定
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    # utils フォルダがスクリプトと同じ階層にある場合
    sys.path.append(os.path.join(BASE_DIR)) 
    # utils フォルダが1つ上の階層にある場合
    sys.path.append(os.path.dirname(BASE_DIR)) # ★ 念のため親も追加
    
    # スクリプトとして実行する場合、相対インポート '.' は使えないため
    # 'from utils.data_parser import DataParser' のような形を試す
    try:
        from utils.data_parser import DataParser
    except ImportError:
        # utils がない場合、同じ階層にあると仮定
        from data_parser import DataParser
        
    import open3d as o3d # DataParser が open3d を使うため
except ImportError as e:
    print(f"エラー: DataParser または open3d のインポートに失敗しました: {e}")
    print("DataParser のパスが正しいか (utils/data_parser.py)、open3d がインストールされているか確認してください (pip install open3d)")
    sys.exit(1)

# あなたが持っている .ply ファイルなどが保存されている親ディレクトリ
RAW_DATA_DIR = '/home/ubuntu/slocal1/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds/dataset/Scenefound3D/'

# 訓練用シーンのIDリスト (例: 420673) が書かれたテキストファイル
TRAIN_SCENES_LIST_FILE = os.path.join(RAW_DATA_DIR, 'train_scenes.txt') # ファイル名のみを指定

# 検証用シーンのIDリストが書かれたテキストファイル
VAL_SCENES_LIST_FILE = os.path.join(RAW_DATA_DIR, 'val_scenes.txt') # ファイル名のみを指定

# .h5 ファイルの出力先ディレクトリ
OUTPUT_DIR = '/home/ubuntu/slocal1/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds/data/scenefun3D_processed_h5'

# 1つのブロック（.h5ファイル）に保存する点の数
NUM_POINTS_PER_BLOCK = 4096

# 1つのシーンから何ブロックサンプリングするか
BLOCKS_PER_SCENE = 100 

# ---------------------------------------------------------------------
# --- [修正] クラス名とIDのマッピング (アフォーダンス用に変更) ---
# SceneFun3D のアフォーダンスラベルに対応させます
CLASS_NAME_TO_ID = {
    # ID 0: アフォーダンスなし (背景 / アノテーションされていない点)
    'background': 0, 
    
    # 警告で報告されたアフォーダンスラベル (ID: 1から割り当て)
    'rotate': 1,
    'key_press': 2,
    'pinch_pull': 3,
    'tip_push': 4,
    'unplug': 5,
    'hook_turn': 6,
    'plug_in': 7,
    'hook_pull': 8,

    # ★ 2025/10/20 修正: 'foot_push' を追加
    'foot_push' : 9,

    'exclude': 255, 
}
# デフォルトIDを 'background' (アフォーダンスなし) の 0 に設定
DEFAULT_LABEL_ID = 0 
# ---------------------------------------------------------------------


def read_scene_data_with_parser(scene_id, data_parser): # 関数名を変更、引数に data_parser を追加
    """
    指定された scene_id のデータを DataParser を使って読み込む
    1. レーザースキャン (.ply) から点群 (XYZRGB) を取得
    2. アノテーション (.json) からラベルを取得
    """
    
    # 1. 点群 (XYZRGB) の読み込み
    try:
        # get_laser_scan() は Open3D の PointCloud オブジェクトを返す
        pcd = data_parser.get_laser_scan(visit_id=scene_id) 
        
        # NumPy 配列に変換
        points_xyz = np.asarray(pcd.points, dtype=np.float32)
        # 色情報も NumPy 配列に変換 (0-1 の範囲のはず)
        colors_rgb = np.asarray(pcd.colors, dtype=np.float32)
        
        # XYZ と RGB を結合 (N, 6)
        if colors_rgb.shape[0] == points_xyz.shape[0]:
             points_data = np.hstack((points_xyz, colors_rgb))
        else:
             print(f"警告: シーン {scene_id} の色情報が点数と一致しません ({colors_rgb.shape[0]} vs {points_xyz.shape[0]})。XYZのみ使用します。")
             # 色がない場合は 0 で埋める (N, 6)
             points_data = np.hstack((points_xyz, np.zeros((points_xyz.shape[0], 3), dtype=np.float32)))

    except FileNotFoundError:
        print(f"警告: シーン {scene_id} のレーザースキャンファイルが見つかりません。スキップします。")
        return None, None
    except Exception as e:
        print(f"エラー: シーン {scene_id} のレーザースキャン読み込みに失敗: {e}")
        return None, None

    num_points = points_data.shape[0] # シーンの総点数 N
    if num_points == 0:
        print(f"警告: シーン {scene_id} に点が含まれていません。スキップします。")
        return None, None

    # 2. アノテーション (ラベル) の読み込み
    try:
        # group_excluded_points=True で 'exclude' は1つにまとめられる
        annotations = data_parser.get_annotations(visit_id=scene_id, group_excluded_points=True)
    except FileNotFoundError:
         print(f"警告: シーン {scene_id} のアノテーションファイル (.json) が見つかりません。")
         # ラベルなしで処理を進めるか、エラーにするか選択
         # ここではラベル不明 (-1 や DEFAULT_LABEL_ID) で埋めて進める例
         print(f"       全ての点をデフォルトID ({DEFAULT_LABEL_ID}) として処理します。")
         labels_data = np.full(num_points, DEFAULT_LABEL_ID, dtype=np.int32)
         # または return None, None でスキップする
         # return None, None
    except Exception as e:
        print(f"エラー: シーン {scene_id} のアノテーション読み込みに失敗: {e}")
        return None, None

    # 3. ラベル配列の作成 (N,) - アノテーションファイルが見つかった場合
    if 'annotations' in locals(): # annotations 変数が存在する場合のみ実行
        # まず、全ての点をデフォルトID (0: 'background') で初期化
        labels_data = np.full(num_points, DEFAULT_LABEL_ID, dtype=np.int32) 

        # 4. アノテーション情報を使ってラベル配列を上書き
        for ann in annotations:
            label_name = ann['label']
            indices = ann['indices']
            
            if not indices: # indices が空リストの場合スキップ
                continue

            # ★ 修正: マッピングに 'label_name' があるかチェック
            if label_name in CLASS_NAME_TO_ID:
                label_id = CLASS_NAME_TO_ID[label_name]
                
                # indices 配列に含まれるインデックスのラベルを上書き
                try:
                    # indices はリストなので NumPy 配列に変換
                    indices_np = np.array(indices, dtype=np.int64) # 点のインデックスは大きい可能性
                    
                    # 安全のため、インデックスが点の数を超えていないかチェック
                    valid_mask = indices_np < num_points
                    valid_indices = indices_np[valid_mask]
                    
                    if not np.all(valid_mask):
                        print(f"警告: シーン {scene_id}, ラベル '{label_name}' に無効なインデックス ({np.sum(~valid_mask)}個) が含まれます。無視します。")

                    if valid_indices.size > 0:
                        labels_data[valid_indices] = label_id
                    else:
                         print(f"警告: シーン {scene_id}, ラベル '{label_name}' の有効なインデックスがありません。")

                except IndexError:
                     print(f"エラー: シーン {scene_id}, ラベル '{label_name}' でインデックスエラーが発生しました。最大インデックス: {np.max(indices_np)}, 点数: {num_points}")
                except Exception as e:
                     print(f"エラー: シーン {scene_id}, ラベル '{label_name}' の処理中に予期せぬエラー: {e}")

            else:
                # ★ 'background' や 'floor' など、意図的に除外したラベルでないか確認
                # (今回は 'background' は 0 なのでここには来ないはず)
                print(f"警告: シーン {scene_id} に未知のラベル名 '{label_name}' があります。無視します。（CLASS_NAME_TO_ID を確認してください）")

    # 5. 点群データとラベルデータを返す
    return points_data, labels_data

# --- [関数維持] sample_blocks 関数は変更なし ---
def sample_blocks(points, labels, num_points_per_block, num_blocks):
    """
    シーン全体 (N, C) から (num_blocks, num_points_per_block, C) をサンプリングする
    (これは単純なランダムサンプリングです)
    """
    N = points.shape[0] # シーンの総点数
    C = points.shape[1] # 特徴量の次元数 (6のはず)
    
    data_blocks = np.zeros((num_blocks, num_points_per_block, C), dtype=np.float32)
    label_blocks = np.zeros((num_blocks, num_points_per_block), dtype=np.int32)
    
    if N == 0:
        # 点がない場合は 0 で埋まった配列を返す
        print("警告: sample_blocks に空の点群が渡されました。")
        return data_blocks, label_blocks

    sampled_count = 0
    attempts = 0
    max_attempts = num_blocks * 5 # 無限ループを防ぐ試行回数上限

    while sampled_count < num_blocks and attempts < max_attempts:
        attempts += 1
        # シーンからランダムにサンプリング開始点（重心）を選ぶ
        # center_idx = np.random.randint(0, N)
        # center_point = points[center_idx, :3] # XYZ座標

        # (より良い方法: 1m x 1m のグリッドなどでサンプリングする)
        # ここでは単純化のため、シーン全体からランダムに N_POINTS を選びます
        
        if N >= num_points_per_block:
            # 点が十分にある場合 -> 重複なしでランダムサンプリング
            indices = np.random.choice(N, num_points_per_block, replace=False)
        else:
            # 点が足りない場合 -> 重複ありでサンプリング
            indices = np.random.choice(N, num_points_per_block, replace=True)
            
        # [追加] サンプリングした点に有効なラベルが含まれているか確認（任意）
        # if np.all(labels[indices] == CLASS_NAME_TO_ID.get('exclude', -1)): # もし exclude ラベルのみならスキップ
        #     continue

        data_blocks[sampled_count] = points[indices]
        label_blocks[sampled_count] = labels[indices]
        sampled_count += 1
        
    if sampled_count < num_blocks:
         print(f"警告: 目標の {num_blocks} ブロックをサンプリングできませんでした ({sampled_count} ブロックのみ)。")
        
    return data_blocks, label_blocks


def save_to_h5(h5_filename, data_blocks, label_blocks):
    """
    (B, N, C) のデータを .h5 ファイルに保存する
    """
    try:
        with h5py.File(h5_filename, 'w') as f:
            f.create_dataset('data', data=data_blocks, dtype='float32', 
                             compression="gzip", compression_opts=4) # 圧縮を追加
            f.create_dataset('label', data=label_blocks, dtype='int32',
                             compression="gzip", compression_opts=1) # 圧縮を追加
    except Exception as e:
         print(f"エラー: H5 ファイル '{h5_filename}' の保存に失敗: {e}")

# --- [関数修正] process_scene 関数を修正 ---
def process_scene(scene_id, output_dir, file_list_txt, data_parser): # ★ 引数に data_parser を追加
    """
    単一のシーンを処理する
    1. DataParser で読み込み
    2. ブロックにサンプリング
    3. .h5 に保存
    """
    
    # 1. 読み込み (★ 修正した関数を呼び出し、data_parser を渡す)
    points, labels = read_scene_data_with_parser(scene_id, data_parser) 
    if points is None or labels is None:
        print(f"シーン {scene_id} のデータ読み込みに失敗したため、スキップします。")
        return

    # 2. サンプリング
    data_blocks, label_blocks = sample_blocks(points, labels, 
                                              NUM_POINTS_PER_BLOCK, 
                                              BLOCKS_PER_SCENE)
    
    # sample_blocks が目標数に満たない場合でも処理を続ける
    if data_blocks.shape[0] == 0:
        print(f"シーン {scene_id} からブロックをサンプリングできませんでした。スキップします。")
        return

    # 3. H5 に保存
    h5_filename = os.path.join(output_dir, f'{scene_id}.h5')
    save_to_h5(h5_filename, data_blocks, label_blocks)
    
    # 4. ファイルリストに追記
    try:
        # (data.data_root からの相対パスを書き込む)
        # OUTPUT_DIR を基準ディレクトリとする
        relative_path = os.path.relpath(h5_filename, output_dir) 
        
        with open(file_list_txt, 'a') as f:
            f.write(relative_path + '\n')
    except Exception as e:
         print(f"エラー: ファイルリスト '{file_list_txt}' への書き込みに失敗: {e}")


# --- [関数修正] main 関数を修正 ---
def main():
    """
    メインの前処理実行関数
    """
    print(f"出力先ディレクトリを作成します: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ★ DataParser をインスタンス化 (データルートを渡す)
    try:
        # data_root_path は dataset ID が含まれるディレクトリ (Scenefound3D の親) を期待する可能性あり
        # DataParser の実装を確認する必要がある
        
        # --- ここを修正 (前回修正済み) ---
        # DataParser は 'Scenefound3D' フォルダの *親* ディレクトリを data_root_path として期待している可能性が高い
        
        # 修正前: data_parser_root = RAW_DATA_DIR 
        # (RAW_DATA_DIR は '.../Scenefound3D/')
        
        # 修正後: RAW_DATA_DIR の親ディレクトリ ('.../dataset/') を指定する
        data_parser_root = os.path.dirname(RAW_DATA_DIR.rstrip('/')) # 末尾のスラッシュを削除してから親を取得
        
        print(f"DataParser を初期化中 (Root: {data_parser_root})...")
        data_parser = DataParser(data_root_path=data_parser_root)
        print("DataParser 初期化完了。")
        
    except NameError:
         print(f"エラー: DataParser クラスが見つかりません。インポートを確認してください。")
         return
    except ImportError as e:
         print(f"エラー: DataParser のインポートに必要なライブラリが不足しています: {e}")
         print("open3d などをインストールしてください (pip install open3d)")
         return
    except Exception as e:
         print(f"エラー: DataParser の初期化に失敗しました: {e}")
         return
    
    # --- ファイルリスト (.txt) の準備 ---
    train_files_txt = os.path.join(OUTPUT_DIR, 'train_files.txt')
    val_files_txt = os.path.join(OUTPUT_DIR, 'val_files.txt')
    
    # 既存のファイルをクリア
    try:
        if os.path.exists(train_files_txt): os.remove(train_files_txt)
        if os.path.exists(val_files_txt): os.remove(val_files_txt)
    except OSError as e:
         print(f"警告: 既存のファイルリストの削除に失敗: {e}")
    
    # --- 訓練データの処理 ---
    print("\n--- 訓練データの処理開始 ---")
    if not os.path.exists(TRAIN_SCENES_LIST_FILE):
        print(f"エラー: 訓練シーンリスト {TRAIN_SCENES_LIST_FILE} が見つかりません。")
        return
        
    try:
        with open(TRAIN_SCENES_LIST_FILE, 'r') as f:
            train_scene_ids = [line.strip() for line in f if line.strip()]
        print(f"{len(train_scene_ids)} 個の訓練シーンを処理します。")
    except Exception as e:
        print(f"エラー: 訓練シーンリスト {TRAIN_SCENES_LIST_FILE} の読み込みに失敗: {e}")
        return

    for scene_id in tqdm.tqdm(train_scene_ids, desc="Train Scenes"):
        # ★ process_scene に data_parser を渡す
        process_scene(scene_id, OUTPUT_DIR, train_files_txt, data_parser) 

    # --- 検証データの処理 ---
    print("\n--- 検証データの処理開始 ---")
    if not os.path.exists(VAL_SCENES_LIST_FILE):
        print(f"エラー: 検証シーンリスト {VAL_SCENES_LIST_FILE} が見つかりません。")
        return

    try:
        with open(VAL_SCENES_LIST_FILE, 'r') as f:
            val_scene_ids = [line.strip() for line in f if line.strip()]
        print(f"{len(val_scene_ids)} 個の検証シーンを処理します。")
    except Exception as e:
        print(f"エラー: 検証シーンリスト {VAL_SCENES_LIST_FILE} の読み込みに失敗: {e}")
        return

    for scene_id in tqdm.tqdm(val_scene_ids, desc="Val Scenes"):
        # ★ process_scene に data_parser を渡す
        process_scene(scene_id, OUTPUT_DIR, val_files_txt, data_parser) 

    print("\n--- 前処理が完了しました ---")
    print(f"H5 ファイルが {OUTPUT_DIR} に保存されました。")
    print(f"訓練リスト: {train_files_txt}")
    print(f"検証リスト: {val_files_txt}")
    print(f"\nconfig ファイル (full_shape_cfg.py) の 'data.data_root' を '{OUTPUT_DIR}' に設定してください。")


if __name__ == "__main__":
    # 必要なライブラリを確認
    try:
        import h5py
        import plyfile # Note: Might not be strictly needed if open3d handles all PLY
        import tqdm
        import open3d
        import json
    except ImportError as e:
        print(f"エラー: 必要なライブラリがありません: {e}")
        print("必要なライブラリ (h5py, plyfile, tqdm, open3d, json) をインストールしてください。")
        print("例: pip install h5py plyfile tqdm open3d")
        sys.exit(1)
        
    main()