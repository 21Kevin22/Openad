"""
3D Point Cloud Part Segmentation Inference Code
(Final Version)
"""
import argparse
import torch
import numpy as np
import open3d as o3d
import importlib
import sys
import os
import json
import matplotlib.pyplot as plt
import pickle

# (seg_classes, cat_to_id, seg_label_to_cat の定義は変更なし)
seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
cat_to_id = {v: i for i, v in enumerate(seg_classes.keys())}
seg_label_to_cat = {}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def parse_args():
    """コマンドライン引数を解析します。"""
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation Inference')
    parser.add_argument('--model', type=str, default='openad_pn2', help='model name')
    parser.add_argument('--weights', type=str, required=True, help='Path to the pretrained weights file (.pth, .t7)')
    parser.add_argument('--input', type=str, required=True, help='Path to the input point cloud file (.ply, .txt, .json, .npy)')
    parser.add_argument('--category', type=str, required=True, help=f'Category of the object. Choices: {list(seg_classes.keys())}')
    parser.add_argument('--npoint', type=int, default=2048, help='Number of points to sample')
    parser.add_argument('--output_file', type=str, default=None, help='(Optional) Path to save the segmented point cloud with labels')
    parser.add_argument('--no_visualize', action='store_true', help='Disable interactive 3D visualization')
    parser.add_argument('--output_image', type=str, default=None, help='(Optional) Path to save a 2D image of the visualization')
    parser.add_argument('--k', type=int, default=20, help='Dummy argument for model compatibility')
    return parser.parse_args()

def to_categorical(y, num_classes):
    """数値をワンホットエンコーディングベクトルに変換します。"""
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if y.is_cuda:
        return new_y.cuda()
    return new_y

def load_point_cloud(file_path, n_points, category):
    file_extension = os.path.splitext(file_path)[1]
    points = np.array([])

    if file_extension == '.pkl':
        # ...既存のpkl処理...
        pass
    elif file_extension == '.json':
        # ...既存のjson処理...
        pass
    elif file_extension == '.ply':
        print(f"Parsing PLY file: {file_path}")
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points).astype(np.float32)
    # 他の形式...

    if len(points) == 0:
        raise ValueError(f"Failed to load any valid points from the file: {file_path}")

    print(f"Loaded {len(points)} points from {file_path}")

    if len(points) > n_points:
        choice = np.random.choice(len(points), n_points, replace=False)
        points = points[choice, :]
    else:
        choice = np.random.choice(len(points), n_points, replace=True)
        points = points[choice, :]

    points = points - np.mean(points, axis=0)
    dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    if dist > 0:
        points = points / dist

    return points

def get_affordance_colors(affordances):
    """アフォーダンスごとに色を割り当てる（RGB値）"""
    cmap = plt.get_cmap('tab10')
    color_map = {}
    for idx, aff in enumerate(affordances):
        color_map[idx] = cmap(idx % 10)[:3]
    return color_map

def visualize_segmentation(points, labels, affordances=None):
    """セグメンテーション結果を3Dでインタラクティブに可視化（アフォーダンスごとに色分け）"""
    print("Visualizing segmentation... Press 'q' in the window to close.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if affordances is not None:
        color_map = get_affordance_colors(affordances)
        colors = np.array([color_map[label] for label in labels])
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        try:
            cmap = torch.hub.load('matplotlib/matplotlib', 'get_cmap', name='tab20', trust_repo=True)
            colors = cmap(labels % 20)[:, :3]
            pcd.colors = o3d.utility.Vector3dVector(colors)
        except Exception as e:
            print(f"Could not load matplotlib for coloring: {e}. Displaying without colors.")
    o3d.visualization.draw_geometries([pcd])

def save_visualization_image(points, labels, path, affordances=None):
    """Matplotlibでアフォーダンスごとに色分けした画像を保存"""
    print(f"Saving visualization image to {path} using Matplotlib...")
    try:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection='3d')
        if affordances is not None:
            color_map = get_affordance_colors(affordances)
            point_colors = np.array([color_map[label] for label in labels])
        else:
            cmap = plt.get_cmap('tab20')
            point_colors = cmap(labels.flatten() % 20)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=point_colors, s=5)
        ax.set_title('Segmentation Result')
        ax.set_axis_off()
        ax.view_init(elev=20, azim=45)
        plt.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Image saved successfully to {path}")
    except Exception as e:
        print(f"An error occurred while saving the image with Matplotlib: {e}")

def main():
    """メインの推論処理です。"""
    args = parse_args()
    if args.category not in seg_classes:
        print(f"Error: Category '{args.category}' is not supported.")
        print(f"Supported categories are: {list(seg_classes.keys())}")
        return

    print(f"Loading model: {args.model}")
    try:
        MODEL = importlib.import_module(f'models.{args.model}')
        classifier = MODEL.OpenAD_PN2(num_classes=16, args=args).cuda()
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    try:
        checkpoint = torch.load(args.weights, map_location='cuda')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        classifier.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
        
    classifier.eval()

    try:
        points = load_point_cloud(args.input, args.npoint, args.category)
        points_tensor = torch.from_numpy(points).float().unsqueeze(0).cuda()
        points_tensor = points_tensor.transpose(2, 1)
    except Exception as e:
        print(f"Error loading or processing point cloud: {e}")
        return

    with torch.no_grad():
        print("Running inference...")
        affordance_list = {
            'Knife': ['grasp', 'cut', 'contain'],
            'Mug': ['grasp', 'contain', 'drink'],
            'Chair': ['sit', 'grasp', 'move'],
        }
        affordances = affordance_list.get(args.category, ['grasp'])
        seg_pred = classifier(points_tensor, affordances)
        pred_choice = torch.max(seg_pred, dim=1)[1]
        pred_labels = pred_choice.squeeze().cpu().numpy()
        print("Inference complete.")

    # --- 5. 結果の可視化と保存 ---
    if args.output_image:
        save_visualization_image(points, pred_labels, args.output_image, affordances)
    if not args.no_visualize:
        visualize_segmentation(points, pred_labels, affordances)
    if args.output_file:
        try:
            output_data = np.hstack((points, pred_labels.reshape(-1, 1)))
            np.savetxt(args.output_file, output_data, fmt='%.6f %d')
            print(f"Result saved to {args.output_file}")
        except Exception as e:
            print(f"Error saving output file: {e}")

if __name__ == '__main__':
    main()