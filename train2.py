import os
from os.path import join as opj
import torch
import torch.nn as nn
import numpy as np
from gorilla.config import Config
from utils import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Number of gpus to use"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="The checkpoint to be resume"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.work_dir != None:
        cfg.work_dir = args.work_dir
    if args.gpu != None:
        cfg.training_cfg.gpu = args.gpu
        
    logger = IOStream(opj(cfg.work_dir, 'run.log'))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
    num_gpu = len(cfg.training_cfg.gpu.split(','))
    logger.cprint('Use %d GPUs: %s' % (num_gpu, cfg.training_cfg.gpu))
    if cfg.get('seed') != None:
        set_random_seed(cfg.seed)
        logger.cprint('Set seed to %d' % cfg.seed)
    model = build_model(cfg).cuda() 

    if args.checkpoint != None:
        print("Loading checkpoint....")
        _, exten = os.path.splitext(args.checkpoint)
        if exten == '.t7':
            model.load_state_dict(torch.load(args.checkpoint))
        elif exten == '.pth':
            check = torch.load(args.checkpoint)
            model.load_state_dict(check['model_state_dict'])
    else:
        print("Training from scratch!")

    dataset_dict = build_dataset(cfg)
    loader_dict = build_loader(cfg, dataset_dict)
    train_loss = build_loss(cfg)
    optim_dict = build_optimizer(cfg, model)

    training = dict(
        model=model,
        dataset_dict=dataset_dict,
        loader_dict=loader_dict,
        loss=train_loss,
        optim_dict=optim_dict,
        logger=logger
    )

    task_trainer = Trainer(cfg, training)
    task_trainer.run()

    try:
        task_trainer.run()
        
    except Exception as e:
        logger.cprint(f"[ERROR] Unhandled exception in main: {e}")
        import traceback
        logger.cprint(traceback.format_exc()) # スタックトレースをログに出力
        
    finally:
        # --- ここからがクリーンアップ処理 ---
        # task_trainer.run() が正常終了しても、エラーで落ちても、
        # この finally ブロックが実行されます。
        
        logger.cprint("Experiment run finished. Cleaning up resources...")
        
        # GPU メモリを保持している可能性のある主要なオブジェクトを削除
        try:
            del model
            del training
            del task_trainer
            del dataset_dict
            del loader_dict
            del train_loss
            del optim_dict
            logger.cprint("Deleted main objects.")
            
        except NameError as e:
            # オブジェクトの初期化に失敗した場合など
            logger.cprint(f"[WARN] Failed to delete objects (may be uninitialized): {e}")
        except Exception as e:
            logger.cprint(f"[WARN] Error during object deletion: {e}")

        # PyTorch の CUDA キャッシュを強制的にクリア
        torch.cuda.empty_cache()
        logger.cprint("CUDA cache cleared.")
        
        # (wandb.finish() は trainer.py の run() 内部の finally で
        # 既に呼び出されているはずです)
        
        logger.cprint("Cleanup complete. Exiting train.py.")
        # --- クリーンアップ処理ここまで ---