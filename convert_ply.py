# convert_ply.py (本当の最終版)
import open3d as o3d
import numpy as np

# 入力ファイル
input_file = "/home/ubuntu/slocal1/point-e/output_point_cloud.ply"
# 出力ファイル
output_file = "output_point_cloud_ascii.ply"

print(f"'{input_file}' を読み込んでいます...")
pcd = o3d.io.read_point_cloud(input_file)

if not pcd.has_points():
    print("エラー: 点群の読み込みに失敗しました。")
else:
    print("点群を100倍にスケールアップしています...")
    pcd.scale(100, center=pcd.get_center())

    # データ型を float32 (float) に変換します
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points, dtype=np.float32))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors, dtype=np.float32))

    print("点群の法線を計算しています...")
    # ★★★ 修正点: スケールに合わせて半径を 0.1 から 2.0 に変更 ★★★
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals, dtype=np.float32))
    
    print(f"'{output_file}' に最終版の点群を保存しています...")
    o3d.io.write_point_cloud(output_file, pcd, write_ascii=True)
    print("変換が完了しました！")