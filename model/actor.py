import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import rnn


class Actor(nn.Module):
    """
    input_dim: 输入维度，这里等于n_states
    output_dim: 输出维度，这里等于n_actions
    max_action: action的最大值
    """
    def __init__(self, n_states, n_actions, max_action, init_w=3e-3):
        super(Actor, self).__init__()
        # self.input_size = n_states + n_actions
        # self.hidden_size = 128
        # self.num_layers = 2
        # self.num_directions = 1  # 单向LSTM
        # self.batch_size = batch_size
        # self.device = device

        self.lstm = nn.LSTM(21, 128, 2, batch_first=True)  # 处理历史轨迹(input_size, hidden_size, num_layers)
        self.l1 = nn.Linear(n_states, 128)  # 处理当前的状态数据
        self.l2 = nn.Linear(128, 128)  # 处理当前的状态数据
        self.l3 = nn.Linear(128, n_actions)
        self.max_action = max_action

        nn.init.uniform_(self.l2.weight.detach(), a=-init_w, b=init_w)
        nn.init.uniform_(self.l2.bias.detach(), a=-init_w, b=init_w)

    def forward(self, history, state):
        # h0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(self.device)  # (num_layers, batch, output_size)
        # c0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(self.device)  # (num_layers, batch, output_size)
        self.lstm.flatten_parameters()  # 提高显存的利用率和效率
        x1, _ = self.lstm(history)  # output(batch_size, time_step, hidden_size)
        x1, _ = rnn.pad_packed_sequence(x1, batch_first=True)  # 由packedSequence数据转换成tensor

        state = state.unsqueeze(1)
        x2 = F.relu(self.l1(state))
        x2 = F.relu(self.l2(x2))

        x3 = torch.cat([x1, x2], 1)
        action = torch.tanh(self.l3(x3))  # torch.tanh与F.tanh没有区别
        action = action[:, -1, :]

        return self.max_action * action


if __name__ == '__main__':
    actor = Actor(n_states=3 + 1 + 3 + 1 + 13, n_actions=3, max_action=1)
    print(sum(p.numel() for p in actor.parameters() if p.requires_grad))
