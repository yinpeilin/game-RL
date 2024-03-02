import os
import time

ENVS_NUM = 20  # 进程数量
TRAIN_MODEL_STEP = ENVS_NUM * 1 # 多少个游戏step之后进行训练
TQDMSHOWSTEP = TRAIN_MODEL_STEP * 10  # tdqm 滑条更新
TENSORBOARD_WRITE_STEP = TQDMSHOWSTEP * 2# 多少step后写入tensorboard
SAVE_MODEL_STEP = TENSORBOARD_WRITE_STEP * 100  # 多少step进行模型保存
EPS_CLIP_DECAY_STEP = SAVE_MODEL_STEP * 1 # 每隔多少个游戏step进行探索率衰减

STARTSTEP = ENVS_NUM * 200  # 初始进行的游戏步数
TIMESTEP = 1000000000  # 总游戏步数

TARGET_UPDATE_STEP = 100  # 训练多少轮后更新target_net
BATCH_SIZE = 512 # 模型训练batch_size
LEARNING_RATE = 1e-3  # 探索率
GAMMA_RATE = 0.99  # 遗忘率
START_EPS_RATE = 0.1  # 开始的探索率
EPS_DECREASE_RATIO = 0.95 # 探索率衰减速度
FINAL_EPS_RATE = 0.1  # 结束的探
TAU = 0.5  # 软更新参数，当为1时为硬更新
WEIGHT_DECAY = 0.0  # 正则化系数

BUFFER_SIZE = 50000

# 模型设置相关
SEQ_LEN = 1
OBS_SHAPE = {
    "box_observation": (SEQ_LEN, 4),
}
ACTION_NUM = 2


MONITOR_DIR = "result/monitor"
MODEL_SAVE_DIR = "result/model"
TENSORBOARD_SAVE_DIR = "result/log/"+str(time.time_ns())
