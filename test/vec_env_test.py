import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from game.test_game import CartPoleEnv
from game.vec_game import vec_game
if __name__ == '__main__':
    SEQ_LEN = 1
    OBS_SHAPE = {
        "box_observation": (SEQ_LEN, 4),
    }
    
    vec_game = vec_game(2, CartPoleEnv, monitor_file_dir= "result/monitor", obs_shape_dict= OBS_SHAPE
                        )
    vec_game.reset()
    while True:
        vec_game.step([1,0])
        vec_game.render()