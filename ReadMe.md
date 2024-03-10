# DQN的完全实现

DQN是强化学习中经典的离线学习算法，本项目实现了Dueling-DQN和多进程游戏交互，同时默认添加了三种游戏的训练配置。完整代码详见[此处](https://github.com/yinpeilin/game-RL)

## 项目结构

```python
E:.
│  .gitignore
│  ReadMe.md
│  requirements.txt
├─config
│  │  flapper_train_setting.py	flapper_bird的训练配置
│  │  mario_train_setting.py	马里奥的训练配置
│  │  test_train_setting.py		cartpole的训练配置
├─dqn 训练相关的代码
│  │  dqn_trainner.py
│  │  log_wrapper.py
│  │  replay_buffer.py
├─game 不同游戏的实现
│  │  flapper_game.py
│  │  mario_game.py
│  │  test_game.py
│  │  vec_game_wrapper.py  多进程游戏的主要实现代码
├─model 不同游戏的模型架构，model_wrapper 主要是对torch模型的封装
│  │  flapper_model_arch.py
│  │  mario_model_arch.py
│  │  model_wrapper.py
│  │  test_model_arch.py
├─result 所有相关游戏结果都将储存在该位置
│  ├─log tensorboard_log保存 reward和loss
│  │	│  .gitkeep
│  ├─model 所有训练中保存的模型
│  │   	│  .gitkeep
│  └─monitor 每个游戏的监控结果
│      	│  .gitkeep
│
└─test 测试代码
        env_test.py 单环境测试代码
        new_trainTest.py 训练测试代码
        vec_envTest.py 多进程游戏测试
```



## 项目依赖

训练依赖于torch以及tqdm，numpy

如果需要查看tensorboard则需下载tensorboard。

部分游戏render需要使用opencv-python，

游戏全部依赖于gymnasium。

## 实现特性

- vec_game主要由pipe通信完成

  ```python
  def game_worker(game, monitor_file_path, conn2):
      one_game = game(monitor_file_path = monitor_file_path)
      while True:
          msg, arg = conn2.recv()
          if msg == 'step':
              state, reward, done, truncated, info = one_game.step(arg)
              conn2.send((state, reward, done, truncated, info))
          if msg == 'reset_step':
              state, reward, done, truncated, info = one_game.reset_step(arg)
              conn2.send((state, reward, done, truncated, info))
          elif msg == 'reset':
              state, info = one_game.reset(arg)
              conn2.send((state, info))
          elif msg == 'render':
              one_game.render()
  ```

  

- replay_buffer由numpy完成，所以这里会引入一部分时间消耗

```python
def __init__(self, buffer_size, obs_shape_dict):
        self.buffer_size = buffer_size
        self.size = 0
        self.index = 0
        self.obs_shape_dict = deepcopy(obs_shape_dict)
        
        self.states = {}
        self.next_states = {}
        
        for key, value in self.obs_shape_dict.items():
            state_shape = [self.buffer_size]
            state_shape.extend(value)
            self.states[key] = np.zeros(state_shape, dtype = np.float32)
            self.next_states[key] = np.zeros(state_shape, dtype = np.float32)
        
        self.actions = np.zeros((self.buffer_size, ), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size, ), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, ), dtype=np.float32)
        self.truncateds = np.zeros((self.buffer_size, ), dtype=np.float32)
```



### 部分结果

在配置为12th Gen Intel(R) Core(TM) i7-12700H   2.30 GHz 和3070ti的电脑上，

关于cartpole， 十分钟就可以训练出较好结果：

[Pictures/image/202403102237152.gif at main · yinpeilin/Pictures (github.com)](https://github.com/yinpeilin/Pictures/blob/main/image/202403102237152.gif)

（因为opencv颜色通道显示的关系，部分颜色出现错位，但不影响结果）

关于flapper_bird, 基于图像约要三十分钟左右：

[Pictures/image/202403102221708.gif at main · yinpeilin/Pictures (github.com)](https://github.com/yinpeilin/Pictures/blob/main/image/202403102221708.gif)



## 使用方法



如需要使用自己的游戏，需要依照`config`中的文件进行配置，在`game`文件夹下写完完整的游戏，在`model`文件夹下添加模型架构，然后依照`test/new_trainTest.py`进行训练。

## TIPS

因为gym_super_mario_bros是以gym（gymnasium的早期版本，早已经不再维护）来实现的，所以为了适配gymnasium，我们需要一些额外的配置。

- 手动将gym_super_mario_bros库中的所有`import gym` 改为`import gymnasium as gym`
- 因为gym库和gymnasium库相比，step和reset的参数略有不同，我们还要到`D:\anaconda\envs\game_rl\Lib\site-packages\nes_py`(这个目录与你电脑python的配置相关)中`nes_env.py`中做下列修改：

```python
def reset(self, seed=None):
    -->
def reset(self, seed=None, options=None, return_info=None):
    
def step(self, action):
    ...
	return self.screen, reward, self.done, info
	-->
    return self.screen, reward, self.done, False, info
```
## TODO

- [ ] vec_game的协程模式
- [ ] DDPG的完全实现
- [ ] log_wrapper的实现
