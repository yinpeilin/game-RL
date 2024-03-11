import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from config.cbf_train_setting import *
# from config.test_train_setting import *
# from config.flapper_train_setting import *
from dqn.DQN_trainner import DQNTrainer

if __name__ == '__main__':
    dqn_trainer = DQNTrainer(start_eps_rate= START_EPS_RATE,
                            end_eps_rate= FINAL_EPS_RATE,
                            eps_decrease_ratio = EPS_DECREASE_RATIO,
                            eps_clip_step= EPS_CLIP_DECAY_STEP,
                            train_model_step = TRAIN_MODEL_STEP,
                            target_update_step = TARGET_UPDATE_STEP,
                            tqdm_step = TQDM_STEP,
                            tensorboard_step = TENSORBOARD_WRITE_STEP,
                            save_model_step = SAVE_MODEL_STEP,
                            batch_size= BATCH_SIZE,
                            tensorboard_dir= TENSORBOARD_SAVE_DIR)
    dqn_trainer.vec_game_init(nums= ENVS_NUM,
                            game_class=game_class,
                            monitor_file_dir=MONITOR_DIR,
                            obs_shape_dict= OBS_SHAPE)
    dqn_trainer.model_wrapper_init(model_arch= MODEL_CLASS,
                                optimizer= OPTIMIZER,
                                loss_function= LOSS_FUNC,
                                obs_shape_dict= OBS_SHAPE,
                                act_n= ACTION_NUM,
                                learning_rate= LEARNING_RATE,
                                gamma= GAMMA_RATE,
                                weight_decay= WEIGHT_DECAY,
                                tau= TAU,
                                device= DEVICE,
                                model_save_dir= MODEL_SAVE_DIR)
    dqn_trainer.replay_buffer_init(buffer_arch= BUFFER_ARCH,
                                buffer_size= BUFFER_SIZE,
                                obs_shape_dict= OBS_SHAPE)
    dqn_trainer.update_all(total_step=TIMESTEP, need_train= NEED_TRAIN, need_render= NEED_RENDER)
    