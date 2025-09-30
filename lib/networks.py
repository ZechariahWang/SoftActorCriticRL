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
        self.q_neural_network=MLP(state_dim + action_dim, 1, hidden_dims, layer_size, activation, activation_last=nn.Identity)

    def forward(self, state, action):
        x=torch.cat([state, action], dim=-1)
        q_value=self.q_neural_network(x)
        return q_value

# you shouldnt need a value network forsac with this action network? Not entirely sure but will need to add extra logic in training later on
class ActionNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, layer_size, activation=nn.ReLU):
        super(ActionNetwork, self).__init__()
        self.mean_network=MLP(state_dim, action_dim, hidden_dims, layer_size, activation, activation_last=nn.Identity)
        self.standard_deviation_network=MLP(state_dim, action_dim, hidden_dims, layer_size, activation, activation_last=nn.Identity)

    def forward(self, state):
        mean=self.mean_network(state)
        std_log=torch.clamp(self.standard_deviation_network(state), min=-20, max=2)
        std=std_log.exp() 

        normal=torch.distributions.Normal(mean, std)
        z=normal.rsample()
        action=torch.tanh(z)

        log_prob = normal.log_prob(z).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob, mean, std_log

    def act(self, state, deterministic=False):
        mean=self.mean_network(state)
        if deterministic:
            z=mean
        else:
            std=torch.clamp(self.standard_deviation_network(state), min=-20, max=2).exp()
            z=mean+std*torch.randn_like(mean)

        return torch.tanh(z)
    


"""
in training: 
with torch.no_grad():
    next_action, next_log_prob, _, _ = actor(next_state)
    q1_next = q1_target(next_state, next_action)
    q2_next = q2_target(next_state, next_action)
    min_q_next = torch.min(q1_next, q2_next)
    v_next = min_q_next - alpha * next_log_prob.unsqueeze(-1)
    target_q = reward + gamma * (1 - done) * v_next
"""

"""
wheen updateing the actor

new_action, log_prob, _, _ = actor(state)
q1_val = q1(state, new_action)
q2_val = q2(state, new_action)
min_q = torch.min(q1_val, q2_val)
actor_loss = (alpha * log_prob - min_q).mean()
"""


    

