import os
import random
import sys

# --- [要修正] 設定 ---

# 1. あなたの .ply シーンの親ディレクトリ 
# (例: '420673' フォルダが格納されている 'Scenefound3D' フォルダのパス)
RAW_DATA_DIR = '/home/ubuntu/slocal1/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds/dataset/Scenefound3D'

# 2. .txt ファイルの出力先 
# (前処理スクリプト 'preprocess_scenefun3d.py' が参照する場所)
# (RAW_DATA_DIR と同じ場所で良いでしょう)
OUTPUT_DIR = RAW_DATA_DIR 

# 3. 訓練データの割合 (例: 0.8 = 80%)
TRAIN_SPLIT_RATIO = 0.8
# ---

def create_split_files():
    print(f"シーンフォルダを検索しています: {RAW_DATA_DIR}")
    
    try:
        # RAW_DATA_DIR 直下のディレクトリ名（シーンIDと仮定）を取得
        # （ファイルや .txt を除外し、ディレクトリのみを対象とします）
        all_scene_ids = [d for d in os.listdir(RAW_DATA_DIR) 
                         if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
    except FileNotFoundError:
        print(f"エラー: ディレクトリが見つかりません: {RAW_DATA_DIR}")
        print("スクリプト内の 'RAW_DATA_DIR' のパスを修正してください。")
        return
    except Exception as e:
        print(f"エラー: ディレクトリの読み込みに失敗しました: {e}")
        return

    if not all_scene_ids:
        print(f"エラー: {RAW_DATA_DIR} 内にシーンフォルダ（サブディレクトリ）が見つかりませんでした。")
        print("パスが正しいか、データがそのフォルダ直下にあるか確認してください。")
        return

    print(f"合計 {len(all_scene_ids)} シーンが見つかりました。")
    print(f"例: {all_scene_ids[0]}")

    # 2. リストのシャッフルと分割
    random.seed(42) # いつでも同じ分割結果になるよう固定
    random.shuffle(all_scene_ids)

    split_index = int(len(all_scene_ids) * TRAIN_SPLIT_RATIO)
    train_scene_ids = all_scene_ids[:split_index]
    val_scene_ids = all_scene_ids[split_index:]

    print(f"訓練用に {len(train_scene_ids)} シーン、検証用に {len(val_scene_ids)} シーンを分割します。")

    # 3. ファイルへの保存
    os.makedirs(OUTPUT_DIR, exist_ok=True) # 出力先がなければ作成
    
    train_list_path = os.path.join(OUTPUT_DIR, 'train_scenes.txt')
    val_list_path = os.path.join(OUTPUT_DIR, 'val_scenes.txt')

    try:
        with open(train_list_path, 'w') as f:
            for scene_id in train_scene_ids:
                f.write(scene_id + '\n')
        print(f"訓練リストを保存しました: {train_list_path}")

        with open(val_list_path, 'w') as f:
            for scene_id in val_scene_ids:
                f.write(scene_id + '\n')
        print(f"検証リストを保存しました: {val_list_path}")

    except IOError as e:
        print(f"エラー: ファイルの書き込みに失敗しました: {e}")
    except Exception as e:
        print(f"予期せぬエラー: {e}")

    print("\nリストファイルの作成が完了しました。")
    print(f"次に、'preprocess_scenefun3d.py' を実行して .h5 ファイルを作成してください。")

if __name__ == "__main__":
    create_split_files()