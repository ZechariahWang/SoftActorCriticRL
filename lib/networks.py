import torch
from torch import nn as nn

# this class just makes the standard multi-layer perceptron
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, layer_size, activation=nn.ReLU, activation_last = nn.Identity):
        super(MLP, self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.hidden_dims=hidden_dims
        self.layer_size=layer_size
        self.activation=activation
        self.activation_last=activation_last

        self.nnModule=nn.ModuleList()
        if len(hidden_dims)==0: # if the user does not want any hidden layers js use the last activation
            self.nnModule.append(nn.Linear(input_dim,output_dim))
            self.nnModule.append(activation_last())
        else: # otherwise build the MLP
            self.nnModule.append(nn.Linear(input_dim,layer_size))
            self.nnModule.append(activation())
            for h in hidden_dims:
                self.nnModule.append(nn.Linear(layer_size, layer_size))
                self.nnModule.append(activation())
            self.nnModule.append(nn.Linear(layer_size, output_dim))
            self.nnModule.append(activation_last())

    # define the forward pass
    def forward(self, x):
        for layer in self.nnModule:
            x = layer(x)
        return x

 # this is for the qq network, it produces a q value based on the current state and action of the mlp network thingy   
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, layer_size, activation=nn.ReLU):
        super(QNetwork, self).__init__()
        self.q_neural_network = MLP(state_dim + action_dim, 1, hidden_dims, layer_size, activation, activation_last=nn.Identity)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        q_value = self.q_neural_network(x)
        return q_value
