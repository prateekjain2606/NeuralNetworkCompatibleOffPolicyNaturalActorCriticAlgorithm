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

parser = argparse.ArgumentParser(description='AC TD(0)')
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
critic_h_layers_sizes = config['critic']['hidden_layer_neurons']
gamma = config['gamma']
lr_A = config['actor']['learning_rate']
lr_C = config['critic']['learning_rate']
load_A = config['actor']['load']
load_C = config['critic']['load']
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


class Critic(nn.Module):
    def __init__(self, input_size, h_layers_sizes, output_size):
        super(Critic, self).__init__()
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

def lr_scheduler(optimizerA, optimizerC, total_reward):
    for schedule in config['learning_rate_scheduler']['schedule']:
        if total_reward >= schedule[0][0] and total_reward < schedule[0][1]:
            optimizerA.param_groups[0]['lr'] = schedule[1]['lr_A']
            optimizerC.param_groups[0]['lr'] = schedule[1]['lr_C']

def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters(), lr=lr_A)
    optimizerC = optim.Adam(critic.parameters(), lr=lr_C)
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
            
            value = critic(state)

            next_state = torch.FloatTensor(next_state).to(device)
            next_value = critic(next_state)

            if done:
                running_total_reward = total_reward if running_total_reward == 0 else running_total_reward * 0.9 + total_reward * 0.1
                print('Iteration: {}, Current Total Reward: {}, Running Total Reward: {}'.format(iter, total_reward, round(running_total_reward,2)))

                reward_list.append(running_total_reward)

                if max_running_total_reward <= running_total_reward:
                    torch.save(actor, config['actor']['final_save_path'])
                    torch.save(critic, config['critic']['final_save_path'])
                    max_running_total_reward = running_total_reward
                
                error = reward - value
                critic_loss = error.pow(2)
                optimizerC.zero_grad()
                critic_loss.backward()
                optimizerC.step()
                optimizerC.zero_grad()

                value = critic(state)
                next_value = critic(next_state)
                advantage = reward - value
                actor_loss = -log_prob * advantage.detach()
                optimizerA.zero_grad()
                actor_loss.backward()
                optimizerA.step()
                optimizerA.zero_grad()

                if config['learning_rate_scheduler']['required']:
                    lr_scheduler(optimizerA, optimizerC, max_running_total_reward)

                break
            else:
                error = reward + gamma * next_value.detach() - value
                critic_loss = error.pow(2)
                optimizerC.zero_grad()
                critic_loss.backward()
                optimizerC.step()
                optimizerC.zero_grad()

                value = critic(state)
                next_value = critic(next_state)
                advantage = reward + gamma * next_value.detach() - value
                actor_loss = -log_prob * advantage.detach()
                optimizerA.zero_grad()
                actor_loss.backward()
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
    
    if load_C:
        path_C = config['critic']['load_path']
        critic = torch.load(path_C).to(device)
        print('Critic Model loaded')
    else:    
        critic = Critic(state_size, critic_h_layers_sizes, 1).to(device)
        torch.save(critic, config['critic']['initial_save_path'])

    trainIters(actor, critic, n_iters=iterations)