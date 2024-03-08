import os
import time
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

'''
the basic setting
'''
train = True
render = False

'''
the basic model train setting
'''
ENVS_NUM = 10 # 进程数量
TRAIN_MODEL_STEP = ENVS_NUM * 4 # 多少个游戏step之后进行训练
TQDMSHOWSTEP = TRAIN_MODEL_STEP * 10  # tdqm 滑条更新
TENSORBOARD_WRITE_STEP = TQDMSHOWSTEP * 2# 多少step后写入tensorboard

SAVE_MODEL_STEP = TENSORBOARD_WRITE_STEP * 100  # 多少step进行模型保存
EPS_CLIP_DECAY_STEP = SAVE_MODEL_STEP * 1 # 每隔多少个游戏step进行探索率衰减

STARTSTEP = ENVS_NUM * 300  # 初始进行的游戏步数
TIMESTEP = 10000000  # 总游戏步数

TARGET_UPDATE_STEP = 400  # 训练多少轮后更新target_net
BATCH_SIZE = 1024 # 模型训练batch_size
LEARNING_RATE = 1e-4  # 探索率
GAMMA_RATE = 0.99  # 遗忘率
START_EPS_RATE = 0.1  # 开始的探索率
EPS_DECREASE_RATIO = 0.95 # 探索率衰减速度
FINAL_EPS_RATE = 0.01  # 结束的探
TAU = 0.5  # 软更新参数，当为1时为硬更新

WEIGHT_DECAY = 1e-8  # 正则化系数
BUFFER_SIZE = 50000

MONITOR_DIR = "result/monitor"+str(time.time_ns())
MODEL_SAVE_DIR = "result/model"
TENSORBOARD_SAVE_DIR = "result/log/"+str(time.time_ns())

'''
the setting that you need to load from the game
'''
SEQ_LEN = 4
OBS_SHAPE = {
    'image': (SEQ_LEN, 100, 100),
    "tick": (SEQ_LEN, 1),
    "last_press": (SEQ_LEN, len(COMPLEX_MOVEMENT))
}

ACTION_NUM = len(COMPLEX_MOVEMENT)

