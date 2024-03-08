import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.flapper_train_setting import *
import torch
from dqn.DQN_trainner import DQNTrainer
from dqn.replay_buffer import ReplayBuffer
from model_arch.flapper_model_arch import DuelingDqnNet
from game.flapper_game import FlapperEnv
from game.vec_game import vec_game
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

torch.set_printoptions(precision=4)

if __name__ == '__main__':
    # model初始化9
    model = DQNTrainer(ENVS_NUM, OBS_SHAPE, ACTION_NUM, DuelingDqnNet, ReplayBuffer, eps_clip=START_EPS_RATE,gamma=GAMMA_RATE,
                    lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY, buffer_size = BUFFER_SIZE, target_update=TARGET_UPDATE_STEP,model_save_dir= MODEL_SAVE_DIR)
    model.load_newest(MODEL_SAVE_DIR)
    # 多进程调用 环境初始化
    env_list = vec_game(ENVS_NUM, FlapperEnv, monitor_file_dir=MONITOR_DIR, obs_shape_dict= OBS_SHAPE)
    
    states, __ = env_list.reset()
    writer = SummaryWriter(TENSORBOARD_SAVE_DIR)
    
    with tqdm(total=TIMESTEP) as t:
        # 初始化
        for i in range(0, STARTSTEP, ENVS_NUM):
            actions = model.choose_action(states=states)
            next_states, rewards, dones, truncateds, __= env_list.step(actions)
            model.store(states, actions, rewards, next_states, dones, truncateds)
            states = next_states
            t.update(ENVS_NUM)
        # 训练代码
        tensorboard_loss_all = 0.0
        tensorboard_reward_all = 0.0
        loss_all = 0.0
        reward_all = 0.0

        assert TQDMSHOWSTEP % ENVS_NUM == 0, "TQDMSHOWSTEP should % NUM_ENVS == 0"

        for i in range(STARTSTEP, TIMESTEP, ENVS_NUM):
            actions = model.choose_action(states=states)
            next_states, rewards, dones, truncateds, infos = env_list.step(actions)
            env_list.render()
            model.store(states, actions, rewards, next_states, dones, truncateds)
            states = next_states
            reward_all += sum(rewards)

            if i % TRAIN_MODEL_STEP == 0:
                loss = model.learn(BATCH_SIZE)
                loss_all += loss
                if i % TQDMSHOWSTEP == 0:
                    t.update(TQDMSHOWSTEP)
                    t.set_postfix(loss=loss_all / TQDMSHOWSTEP,
                                reward=reward_all / (TQDMSHOWSTEP), eps=model.eps_clip)
                    tensorboard_loss_all += loss_all
                    tensorboard_reward_all += reward_all
                    loss_all = 0.0
                    reward_all = 0.0
                    if i % TENSORBOARD_WRITE_STEP == 0:
                        writer.add_scalar(
                            "loss_mean", tensorboard_loss_all / (TENSORBOARD_WRITE_STEP), i)
                        writer.add_scalar(
                            "reward_mean", tensorboard_reward_all / (TENSORBOARD_WRITE_STEP), i)
                        writer.add_scalar("eps_clip", model.eps_clip, i)
                        tensorboard_loss_all = 0.0
                        tensorboard_reward_all = 0.0
                        if i % SAVE_MODEL_STEP == 0:
                            model.save()
                            if i % EPS_CLIP_DECAY_STEP == 0:
                                model.eps_clip *= EPS_DECREASE_RATIO
                                if model.eps_clip < FINAL_EPS_RATE:
                                    model.eps_clip = FINAL_EPS_RATE
    writer.close()