import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# GPU设置
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# 超参数
BATCH_SIZE = 60                                 # 样本数量
LR = 0.01                                       # 学习率
EPSILON = 0.9                                   # greedy policy
GAMMA = 0.9                                     # reward discount
TARGET_REPLACE_ITER = 100                       # 目标网络更新频率(固定不懂的Q网络)
MEMORY_CAPACITY = 500                          # 记忆库容量
# 和环境相关的参数
env = gym.make("CartPole-v1",render_mode="human").unwrapped         # 使用gym库中的环境：CartPole，且打开封装(若想了解该环境，请自行百度)
N_state = env.observation_space.shape[0]      # 特征数
N_action = env.action_space.n
class Net(nn.Module):
    def __init__(self):

        super(Net,self).__init__()
        self.fc1 = nn.Linear(N_state,50)
        self.fc1.weight.data.normal_(0,0.1)
        self.out = nn.Linear(50,N_action)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        action_value = self.out(x)
        return action_value


# 定义DQN类(定义Q网络以及一个固定的Q网络)
class DQN(object):
    def __init__(self):
        # 创建评估网络和目标网络
        self.eval_net,self.target_net = Net().to(device),Net().to(device)
        self.learn_step_counter = 0  # 学习步数记录
        self.memory_counter = 0      # 记忆量计数
        self.memory = np.zeros((MEMORY_CAPACITY,N_state*2+2)) # 存储空间初始化，每一组的数据为(s_t,a_t,r_t,s_{t+1})
        self.optimazer = torch.optim.Adam(self.eval_net.parameters(),lr=LR)
        self.loss_func = nn.MSELoss()     # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        self.loss_func = self.loss_func.to(device)

    def choose_action(self,x):
        x = torch.unsqueeze(torch.FloatTensor(x),0).to(device)  # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        # 设置探索机制
        if np.random.uniform()< EPSILON:
            # 若小于设定值，则采用Q中的最优方法
            action_value = self.eval_net(x)
            # 选定action
            action = torch.max(action_value,1)[1].data.cpu().numpy() # 输出每一行最大值的索引，并转化为numpy ndarray形式
            action = action[0]
        else:
            action = np.random.randint(0,N_action)

        return action

    def store_transition(self,s,a,r,s_):
        transition = np.hstack((s,[a,r],s_))   # 因为action和reward就只是个值不是列表，所以要在外面套个[]
        # 如果记忆满了需要覆盖旧的数据
        index = self.memory_counter % MEMORY_CAPACITY   # 确定在buffer中的行数
        self.memory[index,:]=transition        # 用新的数据覆盖之前的之前
        self.memory_counter +=1

    def learn(self):
        # 目标网络更新，就是我们固定不动的网络
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())   # 将评价网络的权重参数赋给目标网络
        self.learn_step_counter +=1                 # 目标函数的学习次数+1

        # 抽buffer中的数据学习
        sample_idex = np.random.choice(MEMORY_CAPACITY,BATCH_SIZE)   # 在[0, 2000)内随机抽取32个数，可能会重复,若更改超参数会变更
        b_memory = self.memory[sample_idex,:]    # 抽取选中的行数的数据

        # 抽取出32个s数据，保存入b_s中
        b_s = torch.FloatTensor(b_memory[:,:N_state]).to(device)
        # 抽取出32个a数据，保存入b_a中
        b_a = torch.LongTensor(b_memory[:,N_state:N_state+1]).to(device)
        # 抽取出32个r数据，保存入b_r中
        b_r = torch.FloatTensor(b_memory[:,N_state+1:N_state+2]).to(device)
        # 抽取出32个s_数据，保存入b_s_中
        b_s_ = torch.FloatTensor(b_memory[:,-N_state:]).to(device)

        # 获得32个trasition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(b_s).gather(1, b_a)         # 因为已经确定在s时候所走的action，因此选定该action对应的Q值
        # q_next 不进行反向传播，故用detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_next = self.target_net(b_s_).detach()
        # 先算出目标值q_target，max(1)[0]相当于计算出每一行中的最大值（注意不是上面的索引了,而是一个一维张量了），view()函数让其变成(32,1)
        q_target = b_r + GAMMA*q_next.max(1)[0].view(BATCH_SIZE,1)
        # 计算损失值
        loss = self.loss_func(q_eval,q_target)
        self.optimazer.zero_grad()# 清空上一步的残余更新参数值
        loss.backward() # 误差方向传播
        self.optimazer.step() # 逐步的梯度优化

dqn= DQN()

for i in range(400):                    # 设置400个episode
    print(f"<<<<<<<<<第{i}周期")
    s,_ = env.reset()                    # 重置环境
    episode_reward_sum = 0              # 初始化每个周期的reward值

    while True:
        env.render()                    # 开启画面
        a = dqn.choose_action(s)        # 与环境互动选择action
        s_,r,done, info,_= env.step(a)

        # 可以修改reward值让其训练速度加快
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        new_r = r1 + r2

        ########
        dqn.store_transition(s,a,new_r,s_)  # 储存样本
        episode_reward_sum += r

        s = s_                          # 进入下一个状态

        if dqn.memory_counter > MEMORY_CAPACITY:   # 只有在buffer中存满了数据才会学习
            dqn.learn()

        if done:
            print(f"episode:{i},reward_sum:{episode_reward_sum}")

            break