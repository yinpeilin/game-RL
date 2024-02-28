import os
import time
from gym_super_mario_bros.actions import RIGHT_EASY


NUM_ENVS = 1  # 进程数量

TQDMSHOWSTEP = 200  # tdqm 滑条更新
STARTSTEP = 1000  # 初始进行的游戏步数
TIMESTEP = 10000000  # 总游戏步数
TRAIN_MODEL_STEP = 100  # 多少个游戏step之后进行训练
SAVE_MODEL_STEP = 40000  # 多少step进行模型保存
TENSORBOARD_WRITE_STEP = 20000  # 多少step后写入tensorboard
TARGET_UPDATE_STEP = 100  # 训练多少轮后更新target_net
BATCH_SIZE = 512  # 模型训练batch_size
LEARNING_RATE = 1e-4  # 探索率
GAMMA_RATE = 0.80  # 遗忘率
START_EPS_RATE = 0.25  # 开始的探索率
EPS_DECREASE_RATIO = 0.99 # 探索率衰减速度
FINAL_EPS_RATE = 0.10  # 结束的探
TAU = 1.0  # 软更新参数，当为1时为硬更新
EPS_CLIP_DECAY_STEP = 40000  # 每隔多少个游戏step进行探索率衰减
# WEIGHT_DECAY = 1e-7  # 正则化系数

# 模型设置相关
SEQ_LEN = 4
OBS_SHAPE = {
    'image': (SEQ_LEN, 100,100),
    "tick": (SEQ_LEN, 1),
    "last_press": (SEQ_LEN, len(RIGHT_EASY))
}

ACTION_NUM = len(RIGHT_EASY)

# game_draw
UI_ZOOM = 1.0
GAME_MAP_SIZE = (-300, 300, -300, 300)
GAME_MAP_WATCH_SCALE = (-300, 300, -300, 300)  # (xmin, xmax, ymin, ymax)
CONTROL_CHARCTER_ID = -1
CONTROL_TEAM = 4
ENEMY_TEAM = 5
HERO_SIZE = (6, 4)
SOLIDER_SIZE = (3, 2)
STRAGEGICPOINT_SIZE = (20, 20)

ALL_MAP_IMAGE_SIZE = (2500, 2500)


# process相关属性
CBW_GRPC_START_PORT = 50050
GAME_SERVER_START_PORT = 10000
# REPLAY_SERVER_START_PORT = 15070
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), r"result/model")
SERVER_LOG_DIR = os.path.join(os.path.dirname(__file__), r"server_log")
SCENARIO_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), r"scenario_901100071.cfg")
BOTX_CONFIG_PATH = os.path.join(os.path.dirname(
    __file__), r"cbf_botx_config_1v1_20231205.lua")
REWARD_SAVE_DIR = os.path.join(os.path.dirname(__file__), r"monitor")
TENSORBOARD_SAVE_DIR = os.path.join(os.path.dirname(
    __file__), r"result/log/"+str(time.time_ns())+r"/")
LEVEL_DID = "901000315"


# HERO_IMAGE_PATH = os.path.join(os.path.dirname(__file__), r"../../res/hero.png")
# SOLIDER_IMAGE_PATH = os.path.join(os.path.dirname(__file__), r"../../res/solider.png")
# STRAGEGICPOINT_IMAGE_PATH = os.path.join(os.path.dirname(__file__), r"../../res/strategic.png")

MAP_IMAGE_PATH = os.path.join(os.path.dirname(
    __file__), r"../../res/901100071_background.jpg")
CONTROL_TEAM_COLOR = (255, 255, 255)
CONTROL_MAIN_HERO_COLOR = (255, 255, 255)
ENEMY_TEAM_COLOR = (255, 255, 255)

# the server path
if os.name == 'nt':
    BIN_DIR = r"E:/perforce/cbw/dev/wolfgang/_games/proven_ground/_bin"
    GAME_SERVER_PATH = r"server/proven_ground_game_server/_temp/x64/Profile/proven_ground_game_server.exe"
    PROGRAM_PATH = os.path.join(BIN_DIR, GAME_SERVER_PATH)

    SHOW_WINDOW = True
else:

    BIN_DIR = r"/mnt/exp/cbw/dev/wolfgang/_games/proven_ground/chaos/_source/_engine/bin"
    GAME_SERVER_PATH = r"proven_ground_res_control_server/proven_ground_res_control_server.sh"
    PROGRAM_PATH = os.path.join(BIN_DIR, GAME_SERVER_PATH)

    SHOW_WINDOW = False


# CONFIG_SETTINGS_DICT = {
#     "game_process_setting":
#     {
#         "step_mode": STEP_MODE,
#         "num_env": NUM_ENVS,
#         "program_path": PROGRAM_PATH,
#         "show_window": SHOW_WINDOW,
#         "cbw_grpc_start_port": CBW_GRPC_START_PORT,
#         "game_server_start_port": GAME_SERVER_START_PORT,
#         "model_save_dir": MODEL_SAVE_DIR,
#         "server_log_dir": SERVER_LOG_DIR,
#         "scenario_config_path": SCENARIO_CONFIG_PATH,
#         "botx_config_path": BOTX_CONFIG_PATH,
#         "reward_save_dir": REWARD_SAVE_DIR,
#         "tensorboard_save_dir": TENSORBOARD_SAVE_DIR,
#         "level_did": LEVEL_DID,

#     },
#     "model_setting":
#     {
#         "num_env": NUM_ENVS,
#         "action_shape": ACTION_SHAPE,
#         "obs_shape": OBS_SHAPE,
#         "batch_size": BATCH_SIZE

#     },
#     "resource_setting":
#     {
#         # "hero_image_path":HERO_IMAGE_PATH,
#         # "solider_image_path":SOLIDER_IMAGE_PATH,
#         # "strategicpoint_image_path": STRAGEGICPOINT_IMAGE_PATH,
#         "map_image_path": MAP_IMAGE_PATH,
#         "control_team_color": CONTROL_TEAM_COLOR,
#         "control_character_color": CONTROL_MAIN_HERO_COLOR,
#         "enemy_team_color": ENEMY_TEAM_COLOR,

#         "ui_zoom": UI_ZOOM,
#         "hero_size": HERO_SIZE,
#         "soldier_size": SOLIDER_SIZE,
#         "strategicpoint_size": STRAGEGICPOINT_SIZE,
#         "game_map_size": GAME_MAP_SIZE,
#         "game_map_watch_scale": GAME_MAP_WATCH_SCALE,
#         "all_map_image_size": ALL_MAP_IMAGE_SIZE,
#         "control_character_id": CONTROL_CHARCTER_ID,

#         "control_team": CONTROL_TEAM,
#         "enemy_team": ENEMY_TEAM
#     }
# }
