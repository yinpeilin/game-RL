import os
import time
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

'''
the basic setting
'''
train = False
render = True

'''
the basic model train setting
'''
TIMESTEP = 10000000  # 总游戏步数


MONITOR_DIR = "result/monitor/"+str(time.time_ns())
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

