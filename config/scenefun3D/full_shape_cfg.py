
import os
from os.path import join as opj
from utils import PN2_BNMomentum, PN2_Scheduler

exp_name = "OPENAD_PN2_FULL_SHAPE_Release"
work_dir = opj("./log/scenefun3D", exp_name)
seed = 1
try:
    os.makedirs(work_dir)
except:
    print('Working Dir is already existed!')

scheduler = dict(
    type='lr_lambda',
    lr_lambda=PN2_Scheduler(init_lr=0.001, step=20,
                            decay_rate=0.5, min_lr=1e-5)
)

optimizer = dict(
    type='adam',
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-4
)

model = dict(
    type='scenefun3D_pn2',
    weights_init='scenefun3D_init'
)

training_cfg = dict(
    model=model,
    estimate=True,
    partial=False,
    rotate='None',  # z,so3
    semi=False,
    rotate_type=None,
    batch_size=1,  # メモリ使用量を削減するため1に設定
    gradient_accumulation_steps=4,  # 勾配蓄積でメモリ効率を向上
    epoch=100,
    seed=1,
    dropout=0.5,
    gpu='4',
    ignore_label=-100,  # ignore label for loss calculation
    workflow=dict(
        train=1,
        val=1
    ),
    bn_momentum=PN2_BNMomentum(origin_m=0.1, m_decay=0.5, step=20),
    train_affordance = ['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
               'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
               'listen', 'wear', 'press', 'cut', 'stab', 'none'],
    val_affordance = ['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
               'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
               'listen', 'wear', 'press', 'cut', 'stab', 'none'],
    weights_dir = './data/full_shape_weights.npy'
)

data = dict(
    data_root = '/home/ubuntu/slocal1/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds/data/scenefun3D_processed_h5',
    category = ['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
               'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
               'listen', 'wear', 'press', 'cut', 'stab', 'none']
)
