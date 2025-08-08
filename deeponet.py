import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = torch.tanh

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x


class DeepONet(nn.Module):
    def __init__(self, branch, trunk) -> None:
        super().__init__()
        self.branch = branch
        self.trunk = trunk

    def forward(self, branch_input, trunk_input):
        branch_output = self.branch(branch_input)
        trunk_output = self.trunk(trunk_input)
        output = branch_output @ (trunk_output.T)
        return output


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)
