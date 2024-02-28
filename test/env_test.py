import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from game.test_game import CartPoleEnv
if __name__ == '__main__':
    env = CartPoleEnv(frame_tick=1)
    done = True
    while True:
        if done == True:
            env.reset()
        state, reward, done, truncated, info = env.step(env.action_space.sample())
        env.render()