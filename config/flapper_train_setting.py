import os
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

'''
the basic setting
'''
NEED_TRAIN = True
NEED_RENDER = True                                                                                                                                                                                                                         

'''
the basic model train setting
'''
ENVS_NUM = 1 # 进程数量
TRAIN_MODEL_STEP = 4 # 多少个游戏step之后进行训练
TQDM_STEP = 100  # tdqm 滑条更新
TENSORBOARD_WRITE_STEP = 1000# 多少step后写入tensorboard

SAVE_MODEL_STEP = 20000  # 多少step进行模型保存
EPS_CLIP_DECAY_STEP = 3000 # 每隔多少个游戏step进行探索率衰减

TIMESTEP = 1000000  # 总游戏步数

TARGET_UPDATE_STEP = 400  # 训练多少轮后更新target_net
BATCH_SIZE = 1024 # 模型训练batch_size
LEARNING_RATE = 1e-4  # 探索率
GAMMA_RATE = 0.99  # 遗忘率
START_EPS_RATE = 0.005  # 开始的探索率
EPS_DECREASE_RATIO = 0.95 # 探索率衰减速度
FINAL_EPS_RATE = 0.001  # 结束的探
TAU = 0.5  # 软更新参数，当为1时为硬更新

WEIGHT_DECAY = 1e-8  # 正则化系数

MONITOR_DIR = "result/monitor/"+str(time.time_ns())
MODEL_SAVE_DIR = "result/model"
TENSORBOARD_SAVE_DIR = "result/log/"+str(time.time_ns())

'''
the setting that you need to load from the game
'''

from game.flapper_game import FlapperEnv
game_class = FlapperEnv
SEQ_LEN = 4
OBS_SHAPE = {
    'image': (SEQ_LEN, 100, 100),
    "tick": (SEQ_LEN, 1),
    "last_press": (SEQ_LEN, 2)
}
ACTION_NUM = 2

from model.flapper_model_arch import DuelingDqnNet
MODEL_CLASS = DuelingDqnNet

from dqn.replay_buffer import ReplayBuffer
BUFFER_ARCH = ReplayBuffer
BUFFER_SIZE = 20000


import torch as th

OPTIMIZER = th.optim.Adam
LOSS_FUNC = th.nn.SmoothL1Loss
DEVICE = 'cuda'
