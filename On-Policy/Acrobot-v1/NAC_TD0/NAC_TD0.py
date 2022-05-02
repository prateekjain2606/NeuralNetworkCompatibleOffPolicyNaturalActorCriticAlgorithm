import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='NAC TD(0)')
parser.add_argument('Path',
                       metavar='path',
                       type=str,
                       help='path to configuration file')

args = parser.parse_args()

config_file = args.Path

with open(config_file) as json_file:
    config = json.load(json_file)

def video_callable(episode_number):
    return episode_number%config['recording_frequency'] == 0

env = gym.make(config['env'])
if config['record']:
    env = gym.wrappers.Monitor(env, config['recording_path'], force = True, video_callable=video_callable)

if config['episode_length'] is not None:
    env._max_episode_steps = config['episode_length']

if config['numpy_seed'] is not None:
    np.random.seed(config['numpy_seed'])

if config['environment_seed'] is not None:
    env.seed(config['environment_seed'])

if config['pytorch_seed'] is not None:
    torch.manual_seed(config['pytorch_seed'])

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
actor_h_layers_sizes = config['actor']['hidden_layer_neurons']
v_critic_h_layers_sizes = config['value_critic']['hidden_layer_neurons']
gamma = config['gamma']
lr_A = config['actor']['learning_rate']
lr_A_C = config['advantage_critic']['learning_rate']
lr_V_C = config['value_critic']['learning_rate']
load_A = config['actor']['load']
load_A_C = config['advantage_critic']['load']
load_V_C = config['value_critic']['load']
iterations = config['iterations']

class Actor(nn.Module):
    def __init__(self, input_size, h_layers_sizes, output_size):
        super(Actor, self).__init__()
        self.input_size = input_size
        self.h_layers_sizes = h_layers_sizes
        self.all_layers_sizes = [input_size] + h_layers_sizes + [output_size]
        self.output_size = output_size
        self.linears = nn.ModuleList([nn.Linear(self.all_layers_sizes[i], self.all_layers_sizes[i+1], bias=False) for i in range(len(self.all_layers_sizes)-1)])

    def forward(self, state):
        output = torch.tanh(self.linears[0](state))
        for i in range(1,len(self.linears)-1):
            output = torch.tanh(self.linears[i](output))
        output = self.linears[-1](output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution

class V_Critic(nn.Module):
    def __init__(self, input_size, h_layers_sizes, output_size):
        super(V_Critic, self).__init__()
        self.input_size = input_size
        self.h_layers_sizes = h_layers_sizes
        self.all_layers_sizes = [input_size] + h_layers_sizes + [output_size]
        self.output_size = output_size
        self.linears = nn.ModuleList([nn.Linear(self.all_layers_sizes[i], self.all_layers_sizes[i+1]) for i in range(len(self.all_layers_sizes)-1)])

    def forward(self, state):
        value = F.relu(self.linears[0](state))
        for i in range(1,len(self.linears)-1):
            value = F.relu(self.linears[i](value))
        value = self.linears[-1](value)
        return value

class A_Critic(nn.Module):
    def __init__(self, input_size, output_size):
        super(A_Critic, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(self.input_size, self.output_size, bias=False)

    def forward(self, state):
        value = self.linear1(state)
        return value

def lr_scheduler(optimizerA, optimizerA_C, optimizerV_C, total_reward):
    for schedule in config['learning_rate_scheduler']['schedule']:
        if total_reward >= schedule[0][0] and total_reward < schedule[0][1]:
            optimizerA.param_groups[0]['lr'] = schedule[1]['lr_A']
            optimizerA_C.param_groups[0]['lr'] = schedule[1]['lr_A_C']
            optimizerV_C.param_groups[0]['lr'] = schedule[1]['lr_V_C']

def trainIters(actor, v_critic, a_critic, n_iters):
    optimizerA = optim.Adam(actor.parameters(), lr=lr_A)
    optimizerA_C = optim.Adam(a_critic.parameters(), lr=lr_A_C)
    optimizerV_C = optim.Adam(v_critic.parameters(), lr=lr_V_C)
    running_total_reward = 0
    max_running_total_reward = -float('inf')
    reward_list = []

    for iter in range(n_iters):
        state = env.reset()
        total_reward = 0
        state = torch.FloatTensor(state).to(device)

        for i in count():
            if config['render']:
                env.render()
            
            dist = actor(state)
            
            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            total_reward += reward

            log_prob = dist.log_prob(action).unsqueeze(0)
            pseudo_loss = log_prob
            pseudo_loss.backward(retain_graph=True)

            compatible_features = torch.FloatTensor([]).to(device)
            for params in actor.parameters():
                compatible_features = torch.cat((compatible_features, torch.flatten(params.grad)))

            v_value = v_critic(state)
            a_value = a_critic(compatible_features)

            next_state = torch.FloatTensor(next_state).to(device)
            next_v_value = v_critic(next_state)

            if done:

                running_total_reward = total_reward if running_total_reward == 0 else running_total_reward * 0.9 + total_reward * 0.1
                print('Iteration: {}, Current Total Reward: {}, Running Total Reward: {}'.format(iter, total_reward, round(running_total_reward,2)))

                reward_list.append(running_total_reward)

                if max_running_total_reward <= running_total_reward:
                    torch.save(actor, config['actor']['final_save_path'])
                    torch.save(v_critic, config['value_critic']['final_save_path'])
                    torch.save(a_critic, config['advantage_critic']['final_save_path'])
                    max_running_total_reward = running_total_reward

                delta = reward - v_value
                v_critic_loss = delta.pow(2)
                optimizerV_C.zero_grad()
                v_critic_loss.backward()
                optimizerV_C.step()
                optimizerV_C.zero_grad()

                v_value = v_critic(state)
                next_v_value = v_critic(next_state)
                delta = (reward - v_value).detach()
                error = delta - a_value
                a_critic_loss = error.pow(2)
                optimizerA_C.zero_grad()
                a_critic_loss.backward()
                optimizerA_C.step()
                optimizerA_C.zero_grad()

                optimizerA.zero_grad()
                critic_weights = a_critic.linear1.weight.detach().clone()
                start = 0
                for j in range(0,len(actor.all_layers_sizes)-1):
                    actor.linears[j].weight.grad = -1*torch.reshape(torch.narrow(critic_weights, 1, start, actor.all_layers_sizes[j]*actor.all_layers_sizes[j+1]), (actor.all_layers_sizes[j+1], actor.all_layers_sizes[j]))
                    start += actor.all_layers_sizes[j]*actor.all_layers_sizes[j+1]
                optimizerA.step()
                optimizerA.zero_grad()

                if config['learning_rate_scheduler']['required']:
                    lr_scheduler(optimizerA, optimizerA_C, optimizerV_C, max_running_total_reward)

                break
            else:
                delta = reward + gamma * next_v_value.detach() - v_value
                v_critic_loss = delta.pow(2)
                optimizerV_C.zero_grad()
                v_critic_loss.backward()
                optimizerV_C.step()
                optimizerV_C.zero_grad()

                v_value = v_critic(state)
                next_v_value = v_critic(next_state)
                delta = (reward + gamma * next_v_value - v_value).detach()
                error = delta - a_value
                a_critic_loss = error.pow(2)
                optimizerA_C.zero_grad()
                a_critic_loss.backward()
                optimizerA_C.step()
                optimizerA_C.zero_grad()


                optimizerA.zero_grad()
                critic_weights = a_critic.linear1.weight.detach().clone()
                start = 0
                for j in range(0,len(actor.all_layers_sizes)-1):
                    actor.linears[j].weight.grad = -1*torch.reshape(torch.narrow(critic_weights, 1, start, actor.all_layers_sizes[j]*actor.all_layers_sizes[j+1]), (actor.all_layers_sizes[j+1], actor.all_layers_sizes[j]))
                    start += actor.all_layers_sizes[j]*actor.all_layers_sizes[j+1]
                optimizerA.step()
                optimizerA.zero_grad()

                state = next_state
        
    env.close()
    with open(config['rewards_path'], 'w') as fp:
        json.dump(reward_list, fp, indent=4)



if __name__ == '__main__':
    if load_A:
        path_A = config['actor']['load_path']
        actor = torch.load(path_A).to(device)
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, actor_h_layers_sizes, action_size).to(device)
        torch.save(actor, config['actor']['initial_save_path'])
    
    if load_V_C:
        path_V_C = config['value_critic']['load_path']
        v_critic = torch.load(path_V_C).to(device)
        print('Value Critic Model loaded')
    else:    
        v_critic = V_Critic(state_size, v_critic_h_layers_sizes, 1).to(device)
        torch.save(v_critic, config['value_critic']['initial_save_path'])

    if load_A_C:
        path_A_C = config['advantage_critic']['load_path']
        a_critic = torch.load(path_A_C).to(device)
        print('Advantage Critic Model loaded')
    else:
        a_critic_input_size = state_size*actor_h_layers_sizes[0]
        for i in range(len(actor_h_layers_sizes) - 1):
            a_critic_input_size += actor_h_layers_sizes[i]*actor_h_layers_sizes[i+1]
        a_critic_input_size += actor_h_layers_sizes[-1]*action_size
        a_critic = A_Critic(a_critic_input_size, 1).to(device)
        torch.save(a_critic, config['advantage_critic']['initial_save_path'])

    trainIters(actor, v_critic, a_critic, n_iters=iterations)