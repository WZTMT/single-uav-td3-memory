import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import rnn

'''
当前方案并未改变critic的网络结构，仅评判当前的观测下使用当前的动作的好坏
'''


class Critic(nn.Module):
    def __init__(self, n_states, n_actions, init_w=3e-3):
        super(Critic, self).__init__()

        # Q1 architecture
        self.lstm1 = nn.LSTM(n_states + n_actions, 128, 2, batch_first=True)  # 处理历史轨迹(input_size, hidden_size, num_layers)
        self.l1 = nn.Linear(n_states + n_actions, 128)  # 处理当前的状态数据
        self.l2 = nn.Linear(128, 128)  # 处理当前的状态数据
        self.l3 = nn.Linear(128, 1)

        nn.init.uniform_(self.l3.weight.detach(), a=-init_w, b=init_w)
        nn.init.uniform_(self.l3.bias.detach(), a=-init_w, b=init_w)

        # Q2 architecture
        self.lstm2 = nn.LSTM(n_states + n_actions, 128, 2,
                             batch_first=True)  # 处理历史轨迹(input_size, hidden_size, num_layers)
        self.l4 = nn.Linear(n_states + n_actions, 128)  # 处理当前的状态数据
        self.l5 = nn.Linear(128, 128)  # 处理当前的状态数据
        self.l6 = nn.Linear(128, 1)

        nn.init.uniform_(self.l6.weight.detach(), a=-init_w, b=init_w)
        nn.init.uniform_(self.l6.bias.detach(), a=-init_w, b=init_w)

    def forward(self, history, state, action):
        x = torch.cat([state, action], 1)

        self.lstm1.flatten_parameters()  # 提高显存的利用率和效率
        x1, _ = self.lstm1(history)  # output(batch_size, time_step, hidden_size)
        x1, _ = rnn.pad_packed_sequence(x1, batch_first=True)  # 由packedSequence数据转换成tensor

        x = x.unsqueeze(1)
        x2 = F.relu(self.l1(x))
        x2 = F.relu(self.l2(x2))

        x3 = torch.cat([x1, x2], 1)
        q1 = self.l3(x3)  # torch.tanh与F.tanh没有区别
        q1 = q1[:, -1, :]

        self.lstm2.flatten_parameters()  # 提高显存的利用率和效率
        x4, _ = self.lstm2(history)  # output(batch_size, time_step, hidden_size)
        x4, _ = rnn.pad_packed_sequence(x4, batch_first=True)  # 由packedSequence数据转换成tensor

        x5 = F.relu(self.l4(x))
        x5 = F.relu(self.l5(x5))

        x6 = torch.cat([x4, x5], 1)
        q2 = self.l6(x6)  # torch.tanh与F.tanh没有区别
        q2 = q2[:, -1, :]
        return q1, q2

    def q1(self, history, state, action):
        x = torch.cat([state, action], 1)

        self.lstm1.flatten_parameters()  # 提高显存的利用率和效率
        x1, _ = self.lstm1(history)  # output(batch_size, time_step, hidden_size)
        x1, _ = rnn.pad_packed_sequence(x1, batch_first=True)  # 由packedSequence数据转换成tensor

        x = x.unsqueeze(1)
        x2 = F.relu(self.l1(x))
        x2 = F.relu(self.l2(x2))

        x3 = torch.cat([x1, x2], 1)
        q1 = self.l3(x3)  # torch.tanh与F.tanh没有区别
        q1 = q1[:, -1, :]
        return q1


if __name__ == '__main__':
    actor = Critic(n_states=3 + 1 + 3 + 1 + 13, n_actions=3)
    print(sum(p.numel() for p in actor.parameters() if p.requires_grad))
