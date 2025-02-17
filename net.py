import torch
import torch.nn as nn
import numpy as np


class Actor(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim))

    def forward(self, state):
        # 定义前向传播的方法，将输入 state 通过网络 self.net 得到输出，并应用 tanh 激活函数
        return self.net(state).tanh()  # 使用 tanh 激活函数将输出限制在 [-1, 1] 范围内

    def get_action(self, state, action_std):
        # 定义获取动作的方法，state 是输入状态，action_std 是动作的标准差
        action = self.net(state).tanh()  # 将输入 state 传入神经网络 self.net，得到动作，并应用 tanh 激活函数
        noise = (torch.randn_like(action) * action_std).clamp(-0.5,
                                                              0.5)  # 生成与 action 形状相同的随机噪声，噪声乘以 action_std，并将其限制在 [-0.5, 0.5] 范围内
        return (action + noise).clamp(-1.0, 1.0)  # 将动作和噪声相加，得到最终动作，并将其限制在 [-1, 1] 范围内，然后返回


class Critic(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, 1))

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))  # q value


class CriticTwin(nn.Module):  # shared parameter
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU())  # concat(state, action)
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))  # q2 value

    def forward(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))  # (7,4)二维张量合并成(1,11)
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values