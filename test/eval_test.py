# import os,
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), r".."))
from game.mario_game import mario_env
from DQN_trainner import DQNTrainer
from config.mario_eval_setting import *
import torch
import cv2
import random
# sys.path.append(r'D:\GitCodeSource\minicba\envs\PartialMapDraw\build\Debug')

torch.set_printoptions(precision=2)

if __name__ == '__main__':
    # model初始化
    model = DQNTrainer(OBS_SHAPE, ACTION_NUM, eps_clip=START_EPS_RATE,
                       lr=LEARNING_RATE, target_update=TARGET_UPDATE_STEP, gamma=GAMMA_RATE)
    model.load_newest(MODEL_SAVE_DIR)
    # 多进程调用 环境初始化
    env = mario_env()
    done = True
    all_reward = 0
    while True:
        if done == True:
            state, info = env.reset()
            print(all_reward)
            all_reward = 0.0
        # print(state[0])
        action = model.eval_choose_action([state])
        # print(action)
        # print(action)
        # action = random.randint(0, ACTION_NUM-1)
        print(action)
        state, reward, done, truncated, info = env.step(action[0])
        all_reward += reward
        env.render()
