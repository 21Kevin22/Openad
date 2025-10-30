import pickle
import numpy as np

# 調査したいPickleファイルのパス
file_path = '/home/ubuntu/slocal1/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds/data/full_shape_train_data.pkl'

print(f"Inspecting file: {file_path}")

try:
    with open(file_path, 'rb') as f:
        # encoding='latin1' は古いpickleファイルを読み込む際に必要になることがあります
        data = pickle.load(f, encoding='latin1')

    if isinstance(data, list) and len(data) > 0:
        # --- 1. データセット内の全カテゴリ名を収集 ---
        unique_categories = set()
        for sample in data:
            if isinstance(sample, dict) and 'semantic class' in sample:
                unique_categories.add(sample['semantic class'])
        
        print("\n--- Found Unique Categories ---")
        if unique_categories:
            # アルファベット順にソートして表示
            print(sorted(list(unique_categories)))
        else:
            print("Could not find the 'semantic class' key in any sample.")

        # --- 2. 最初のデータサンプルの詳細な構造を調査 ---
        first_item = data[0]
        print("\n--- Inspecting the first item's detailed structure ---")
        print(f"Top-level keys: {list(first_item.keys())}")

        if 'full_shape' in first_item and isinstance(first_item['full_shape'], dict):
            nested_dict = first_item['full_shape']
            print("\nThe 'full_shape' key contains another dictionary.")
            print(f"Keys INSIDE 'full_shape': {list(nested_dict.keys())}")
            
            # 'full_shape'辞書の中にある点群データらしきものを探す
            for key, value in nested_dict.items():
                if isinstance(value, np.ndarray):
                    print(f"  - Found a NumPy array under key '{key}' with shape: {value.shape}")

except Exception as e:
    print(f"An error occurred: {e}")