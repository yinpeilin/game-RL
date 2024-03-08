import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.mario_eval_setting import *
import torch
from dqn.DQN_trainner import DQNTrainer
from dqn.replay_buffer import ReplayBuffer
from model_arch.mario_model_arch import DuelingDqnNet
from game.mario_game import mario_env
from game.vec_game import vec_game

torch.set_printoptions(precision=4)

if __name__ == '__main__':
    # model初始化9
    model = DQNTrainer(ENVS_NUM, OBS_SHAPE, ACTION_NUM, DuelingDqnNet, ReplayBuffer, eps_clip=START_EPS_RATE,gamma=GAMMA_RATE,
                    lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY, buffer_size = BUFFER_SIZE, target_update=TARGET_UPDATE_STEP,model_save_dir= MODEL_SAVE_DIR)
    model.load_newest(MODEL_SAVE_DIR)
    # 多进程调用 环境初始化
    env_list = vec_game(ENVS_NUM, mario_env, monitor_file_dir=MONITOR_DIR, obs_shape_dict= OBS_SHAPE)
    
    states, __ = env_list.reset()
    
    while True:
        actions = model.choose_action(states=states)
        next_states, rewards, dones, truncateds, infos = env_list.step(actions)
        print(rewards, dones)
        env_list.render()
        states = next_states
