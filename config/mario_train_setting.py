import os
import time
from gym_super_mario_bros.actions import RIGHT_EASY


ENVS_NUM = 10  # 进程数量
TRAIN_MODEL_STEP = ENVS_NUM * 4 # 多少个游戏step之后进行训练
TQDMSHOWSTEP = TRAIN_MODEL_STEP * 10  # tdqm 滑条更新
TENSORBOARD_WRITE_STEP = TQDMSHOWSTEP * 2# 多少step后写入tensorboard

SAVE_MODEL_STEP = TENSORBOARD_WRITE_STEP * 50  # 多少step进行模型保存
EPS_CLIP_DECAY_STEP = SAVE_MODEL_STEP * 1 # 每隔多少个游戏step进行探索率衰减

STARTSTEP = ENVS_NUM * 200  # 初始进行的游戏步数
TIMESTEP = 10000000  # 总游戏步数

TARGET_UPDATE_STEP = 200  # 训练多少轮后更新target_net
BATCH_SIZE = 768 # 模型训练batch_size
LEARNING_RATE = 1e-4  # 探索率
GAMMA_RATE = 0.90  # 遗忘率
START_EPS_RATE = 0.2  # 开始的探索率
EPS_DECREASE_RATIO = 0.95 # 探索率衰减速度
FINAL_EPS_RATE = 0.05  # 结束的探
TAU = 0.5  # 软更新参数，当为1时为硬更新

WEIGHT_DECAY = 0.0  # 正则化系数
BUFFER_SIZE = 50000

# 模型设置相关
SEQ_LEN = 4
OBS_SHAPE = {
    'image': (SEQ_LEN, 100,100),
    "tick": (SEQ_LEN, 1),
    "last_press": (SEQ_LEN, len(RIGHT_EASY))
}

ACTION_NUM = len(RIGHT_EASY)

MONITOR_DIR = "result/monitor"
MODEL_SAVE_DIR = "result/model"
TENSORBOARD_SAVE_DIR = "result/log/"+str(time.time_ns())