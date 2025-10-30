import torch
import numpy as np
from tqdm import tqdm

def evaluation(logger, model, val_loader, affordance):
    num_classes = len(affordance)
    total_correct = 0
    total_seen = 0
    total_seen_class = [0] * num_classes
    total_correct_class = [0] * num_classes
    total_iou_deno_class = [0] * num_classes
    mIoU = 0.0 # mIoU を初期化

    # --- try...finally を削除 ---
    with torch.no_grad(): # 勾配計算は無効化
        model.eval() # モデルを評価モードに

        pbar = tqdm(enumerate(val_loader), total=len(val_loader), smoothing=0.9, desc="Evaluating")

        for i, temp_data in pbar:
            # --- try...finally を削除 ---

            # --- データ準備 ---
            # (エラーハンドリングは削除されるため、ここでエラーが起きると関数が停止します)
            try:
                (data, _, label, _, _) = temp_data
            except ValueError:
                logger.cprint(f"[WARN] Batch {i}: Unexpected data format. Skipping.")
                # --- try...except がないので continue はそのまま使う ---
                continue # 次のバッチへ

            data = data.float().cuda(non_blocking=True)
            label = label.float().cuda(non_blocking=True)
            if label.ndim >= 2: label = torch.squeeze(label)
            if label.ndim == 3 and label.shape[2] == 1: label = label.squeeze(2)
            label = label.long()

            if data.shape[-1] == 3 and len(data.shape) == 3:
                data = data.permute(0, 2, 1).contiguous()
            elif data.shape[1] != 3 or len(data.shape) != 3:
                 logger.cprint(f"[WARN] Batch {i}: Unexpected data shape {data.shape}. Skipping.")
                 continue # 次のバッチへ

            B, C_in, N = data.shape
            if label.shape != (B, N):
                 logger.cprint(f"[WARN] Batch {i}: Label shape mismatch {label.shape}. Expected {(B, N)}. Skipping.")
                 continue # 次のバッチへ

            # --- モデル推論 ---
            # (ここでエラーが起きると関数が停止し、下方の del は実行されません)
            afford_pred = model(data, affordance) # [B, C_out, N]

            # --- 予測ラベル取得 ---
            pred_labels = torch.argmax(afford_pred, dim=1) # [B, N]

            # --- CPU/NumPy 変換 ---
            pred_labels_np = pred_labels.cpu().numpy()
            label_np = label.cpu().numpy()

            # --- メトリクス計算 ---
            # (ここでエラーが起きると関数が停止し、下方の del は実行されません)
            correct = np.sum((pred_labels_np == label_np))
            total_correct += correct
            total_seen += (B * N)

            for class_idx in range(num_classes):
                total_seen_class[class_idx] += np.sum(label_np == class_idx)
                total_correct_class[class_idx] += np.sum((pred_labels_np == class_idx) & (label_np == class_idx))
                total_iou_deno_class[class_idx] += np.sum((pred_labels_np == class_idx) | (label_np == class_idx))


            try:
                del data, label, afford_pred, pred_labels
                del pred_labels_np, label_np
            except NameError:
                 # 変数が定義される前に continue した場合など
                 pass
            except Exception as del_error:
                 logger.cprint(f"[WARN] Batch {i}: Error deleting tensors/arrays at end of loop: {del_error}")

            with np.errstate(divide='ignore', invalid='ignore'):
                iou_class = np.array(total_correct_class) / np.array(total_iou_deno_class, dtype=np.float64)
                acc_class = np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64)

            # ✅ nanmean を使い、分母が全て 0 だった場合も考慮
            mIoU = np.nanmean(iou_class) if np.any(total_iou_deno_class > 0) else 0.0
            mAcc = np.nanmean(acc_class) if np.any(total_seen_class > 0) else 0.0
            accuracy = total_correct / float(total_seen) if total_seen > 0 else 0.0