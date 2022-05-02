import gym, os
from itertools import count, product
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

parser = argparse.ArgumentParser(description='Off-Policy NAC TD(0)')
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
ratio_estimation_env = gym.make(config['env'])

if config['record']:
    env = gym.wrappers.Monitor(env, config['recording_path'], force = True, video_callable=video_callable)

if config['episode_length'] is not None:
    env._max_episode_steps = config['episode_length']
    ratio_estimation_env._max_episode_steps = config['episode_length']

if config['numpy_seed'] is not None:
    np.random.seed(config['numpy_seed'])

if config['environment_seed'] is not None:
    env.seed(config['environment_seed'])
    ratio_estimation_env.seed(config['environment_seed'])

if config['pytorch_seed'] is not None:
    torch.manual_seed(config['pytorch_seed'])

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
actor_h_layers_sizes = config['actor']['hidden_layer_neurons']
v_critic_h_layers_sizes = config['value_critic']['hidden_layer_neurons']
w_h_layers_sizes = config['w']['hidden_layer_neurons']
y_h_layers_sizes = config['y']['hidden_layer_neurons']
gamma = config['gamma']
lr_A = config['actor']['learning_rate']
lr_A_C = config['advantage_critic']['learning_rate']
lr_V_C = config['value_critic']['learning_rate']
lr_W = config['w']['learning_rate']
lr_Y = config['y']['learning_rate']
load_A = config['actor']['load']
load_A_C = config['advantage_critic']['load']
load_V_C = config['value_critic']['load']
load_W = config['w']['load']
load_Y = config['y']['load']
random_behv_prob = config['random_behaviour_probability']
iterations = config['iterations']
estimation_samples = config['estimation_samples']
kernel_sigma = config['kernel_sigma']

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

class W(nn.Module):
    def __init__(self, input_size, h_layers_sizes, output_size):
        super(W, self).__init__()
        self.input_size = input_size
        self.h_layers_sizes = h_layers_sizes
        self.all_layers_sizes = [input_size] + h_layers_sizes + [output_size]
        self.output_size = output_size
        self.linears = nn.ModuleList([nn.Linear(self.all_layers_sizes[i], self.all_layers_sizes[i+1]) for i in range(len(self.all_layers_sizes)-1)])

    def forward(self, state):
        value = F.relu(self.linears[0](state))
        for i in range(1,len(self.linears)-1):
            value = F.relu(self.linears[i](value))
        value = torch.exp(self.linears[-1](value))
        return value

class Y(nn.Module):
    def __init__(self, input_size, h_layers_sizes, output_size):
        super(Y, self).__init__()
        self.input_size = input_size
        self.h_layers_sizes = h_layers_sizes
        self.all_layers_sizes = [input_size] + h_layers_sizes + [output_size]
        self.output_size = output_size
        self.linears = nn.ModuleList([nn.Linear(self.all_layers_sizes[i], self.all_layers_sizes[i+1]) for i in range(len(self.all_layers_sizes)-1)])

    def forward(self, state):
        value = F.relu(self.linears[0](state))
        for i in range(1,len(self.linears)-1):
            value = F.relu(self.linears[i](value))
        value = torch.exp(self.linears[-1](value))
        return value

def kernel(state1, state2, l = 1):
    d = torch.norm(state1 - state2)
    return torch.exp(-((d**2)/(2*(l**2))))

def estimate_W(target_policy, w, optimizerW, random_behv_prob = 1, behaviour_policy = None, iterations = 5000, samples = 1000):

    for iter in range(iterations):
        w_state1_list = []
        w_state2_list = []
        w_next_state1_list = []
        w_next_state2_list = []
        beta1_list = []
        beta2_list = []
        kernel_value_list = []
        z_w_state = 0

        training_data = []
        initial_state = ratio_estimation_env.reset()
        initial_state = torch.FloatTensor(initial_state).to(device)
        state = initial_state

        for i in range(samples):
            
            if random_behv_prob < 1:
                dist_behaviour = behaviour_policy(state)
                d = np.random.uniform()
                if d < random_behv_prob:
                    action = torch.randint(0, action_size, (1,)).to(device)
                    action = torch.squeeze(action)
                else:
                    action = dist_behaviour.sample()
            else:
                action = torch.randint(0, action_size, (1,)).to(device)
                action = torch.squeeze(action)

            next_state, reward, done, _ = ratio_estimation_env.step(action.cpu().numpy())

            next_state = torch.FloatTensor(next_state).to(device)

            dist_target = target_policy(state)

            if random_behv_prob < 1:
                beta = (dist_target.probs[action]/(random_behv_prob*(1/action_size) +(1 - random_behv_prob)*dist_behaviour.probs[action])).detach()
            else:
                beta = dist_target.probs[action].detach()*action_size
            
            training_data.append([state,beta,next_state])

            state = next_state

            if done:
                break

        batch = [[None, None, initial_state]]

        for i in range(len(training_data)):
            d = np.random.uniform()
            if d < gamma**(i+1):
                batch.append(training_data[i])

        pairs = list(product(batch, repeat=2))

        for pair in pairs:
            sample1 = pair[0]
            sample2 = pair[1]

            if sample1[0] != None:
                w_state1 = w(sample1[0])
            else:
                w_state1 = None

            if sample2[0] != None:
                w_state2 = w(sample2[0])
            else:
                w_state2 = None

            beta1 = sample1[1]
            beta2 = sample2[1]

            w_next_state1 = w(sample1[2])
            w_next_state2 = w(sample2[2])

            kernel_value = kernel(sample1[2], sample2[2], kernel_sigma)

            w_state1_list.append(w_state1)
            w_state2_list.append(w_state2)
            w_next_state1_list.append(w_next_state1)
            w_next_state2_list.append(w_next_state2)
            beta1_list.append(beta1)
            beta2_list.append(beta2)
            kernel_value_list.append(kernel_value)

        for sample in batch[1:]:
            w_state = w(sample[0])
            z_w_state += w_state

        z_w_state /= len(batch)

        w_loss = 0

        for i in range(len(pairs)):
            if w_state1_list[i] == None and w_state2_list[i] == None:
                w_loss += (1 - (w_next_state1_list[i]/z_w_state))*(1 - (w_next_state2_list[i]/z_w_state))*kernel_value_list[i]

            elif w_state1_list[i] == None:
                w_loss += (1 - (w_next_state1_list[i]/z_w_state))*(beta2_list[i]*(w_state2_list[i]/z_w_state) - (w_next_state2_list[i]/z_w_state))*kernel_value_list[i]

            elif w_state2_list[i] == None:
                w_loss += (beta1_list[i]*(w_state1_list[i]/z_w_state) - (w_next_state1_list[i]/z_w_state))*(1 - (w_next_state2_list[i]/z_w_state))*kernel_value_list[i]

            else:
                w_loss += (beta1_list[i]*(w_state1_list[i]/z_w_state) - (w_next_state1_list[i]/z_w_state))*(beta2_list[i]*(w_state2_list[i]/z_w_state) - (w_next_state2_list[i]/z_w_state))*kernel_value_list[i]

        w_loss /= len(batch)
        optimizerW.zero_grad()
        w_loss.backward()
        optimizerW.step()
        optimizerW.zero_grad()

def estimate_Y(target_policy, y, optimizerY, random_behv_prob = 1, behaviour_policy = None, iterations = 5000, samples = 1000):

    for iter in range(iterations):
        y_state1_list = []
        y_state2_list = []
        y_next_state1_list = []
        y_next_state2_list = []
        beta1_list = []
        beta2_list = []
        kernel_value_list = []
        z_y_state = 0

        training_data = []
        state = ratio_estimation_env.reset()
        state = torch.FloatTensor(state).to(device)

        for i in range(samples):
            
            if random_behv_prob < 1:
                dist_behaviour = behaviour_policy(state)
                d = np.random.uniform()
                if d < random_behv_prob:
                    action = torch.randint(0, action_size, (1,)).to(device)
                    action = torch.squeeze(action)
                else:
                    action = dist_behaviour.sample()
            else:
                action = torch.randint(0, action_size, (1,)).to(device)
                action = torch.squeeze(action)

            next_state, reward, done, _ = ratio_estimation_env.step(action.cpu().numpy())

            next_state = torch.FloatTensor(next_state).to(device)

            dist_target = target_policy(state)

            if random_behv_prob < 1:
                beta = (dist_target.probs[action]/(random_behv_prob*(1/action_size) +(1 - random_behv_prob)*dist_behaviour.probs[action])).detach()
            else:
                beta = dist_target.probs[action].detach()*action_size
            
            training_data.append([state,beta,next_state])

            state = next_state

            if done:
                break

        pairs = list(product(training_data, repeat=2))

        for pair in pairs:
            sample1 = pair[0]
            sample2 = pair[1]

            y_state1 = y(sample1[0])
            y_state2 = y(sample2[0])

            beta1 = sample1[1]
            beta2 = sample2[1]

            y_next_state1 = y(sample1[2])
            y_next_state2 = y(sample2[2])

            kernel_value = kernel(sample1[2], sample2[2], kernel_sigma)

            y_state1_list.append(y_state1)
            y_state2_list.append(y_state2)
            y_next_state1_list.append(y_next_state1)
            y_next_state2_list.append(y_next_state2)
            beta1_list.append(beta1)
            beta2_list.append(beta2)
            kernel_value_list.append(kernel_value)

        for sample in training_data:
            y_state = y(sample[0])
            z_y_state += y_state

        z_y_state /= len(training_data)

        y_loss = 0

        for i in range(len(pairs)):
            y_loss += (beta1_list[i]*(y_state1_list[i]/z_y_state) - (y_next_state1_list[i]/z_y_state))*(beta2_list[i]*(y_state2_list[i]/z_y_state) - (y_next_state2_list[i]/z_y_state))*kernel_value_list[i]

        y_loss /= len(training_data)
        optimizerY.zero_grad()
        y_loss.backward()
        optimizerY.step()
        optimizerY.zero_grad()

def lr_scheduler(optimizerA, optimizerA_C, optimizerV_C, total_reward):
    for schedule in config['learning_rate_scheduler']['schedule']:
        if total_reward >= schedule[0][0] and total_reward < schedule[0][1]:
            optimizerA.param_groups[0]['lr'] = schedule[1]['lr_A']
            optimizerA_C.param_groups[0]['lr'] = schedule[1]['lr_A_C']
            optimizerV_C.param_groups[0]['lr'] = schedule[1]['lr_V_C']

def evaluate_policy(actor):
    state = env.reset()
    state = torch.FloatTensor(state).to(device)
    total_reward = 0

    for i in count():
        if config['render']:
            env.render()

        dist = actor(state)
        action = dist.sample()

        next_state, reward, done, _ = env.step(action.cpu().numpy())
        total_reward += reward

        next_state = torch.FloatTensor(next_state).to(device)
        state = next_state

        if done:
            break

    return total_reward

def trainIters(actor, v_critic, a_critic, w, y, random_behv_prob = 1, behaviour_policy = None, n_iters = 5000):
    optimizerA = optim.Adam(actor.parameters(), lr=lr_A)
    optimizerA_C = optim.Adam(a_critic.parameters(), lr=lr_A_C)
    optimizerV_C = optim.Adam(v_critic.parameters(), lr=lr_V_C)
    optimizerW = optim.Adam(w.parameters(), lr=lr_W)
    optimizerY = optim.Adam(y.parameters(), lr=lr_Y)
    running_total_reward = 0
    max_running_total_reward = -float('inf')
    reward_list = []

    for iter in range(n_iters):
        state = env.reset()
        state = torch.FloatTensor(state).to(device)
        estimate_W(actor, w, optimizerW, random_behv_prob, behaviour_policy, iterations = 1, samples = estimation_samples)
        estimate_Y(actor, y, optimizerY, random_behv_prob, behaviour_policy, iterations = 1, samples = estimation_samples)
        
        for i in count():
            
            if random_behv_prob < 1:
                dist_behaviour = behaviour_policy(state)
                d = np.random.uniform()
                if d < random_behv_prob:
                    action = torch.randint(0, action_size, (1,)).to(device)
                    action = torch.squeeze(action)
                else:
                    action = dist_behaviour.sample()
            else:
                action = torch.randint(0, action_size, (1,)).to(device)
                action = torch.squeeze(action)          

            dist_target = actor(state)
            
            next_state, reward, done, _ = env.step(action.cpu().numpy())

            if random_behv_prob < 1:
                beta = (dist_target.probs[action]/(random_behv_prob*(1/action_size) +(1 - random_behv_prob)*dist_behaviour.probs[action])).detach()
            else:
                beta = dist_target.probs[action].detach()*action_size
                
            log_prob = dist_target.log_prob(action).unsqueeze(0)
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
                delta = reward - v_value
                v_critic_loss = delta.pow(2)
                optimizerV_C.zero_grad()
                v_critic_loss.backward()
                y_ratio = y(state).detach()
                for j in range(len(v_critic.all_layers_sizes)-1):
                    v_critic.linears[j].weight.grad = y_ratio*beta*v_critic.linears[j].weight.grad
                    v_critic.linears[j].bias.grad = y_ratio*beta*v_critic.linears[j].bias.grad
                optimizerV_C.step()
                optimizerV_C.zero_grad()

                v_value = v_critic(state)
                next_v_value = v_critic(next_state)
                delta = (reward - v_value).detach()
                error = delta - a_value
                a_critic_loss = error.pow(2)
                optimizerA_C.zero_grad()
                a_critic_loss.backward()
                w_ratio = w(state).detach()
                a_critic.linear1.weight.grad = w_ratio*beta*a_critic.linear1.weight.grad
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

                total_reward = evaluate_policy(actor)

                running_total_reward = total_reward if running_total_reward == 0 else running_total_reward * 0.9 + total_reward * 0.1
                print('Iteration: {}, Current Total Reward: {}, Running Total Reward: {}'.format(iter, total_reward, round(running_total_reward,2)))

                reward_list.append(running_total_reward)

                if max_running_total_reward <= running_total_reward:
                    torch.save(actor, config['actor']['final_save_path'])
                    torch.save(v_critic, config['value_critic']['final_save_path'])
                    torch.save(a_critic, config['advantage_critic']['final_save_path'])
                    torch.save(a_critic, config['w']['final_save_path'])
                    torch.save(a_critic, config['y']['final_save_path'])
                    max_running_total_reward = running_total_reward

                if config['learning_rate_scheduler']['required']:
                    lr_scheduler(optimizerA, optimizerA_C, optimizerV_C, max_running_total_reward)

                break
            else:
                delta = reward + gamma * next_v_value.detach() - v_value
                v_critic_loss = delta.pow(2)
                optimizerV_C.zero_grad()
                v_critic_loss.backward()
                y_ratio = y(state).detach()
                for j in range(len(v_critic.all_layers_sizes)-1):
                    v_critic.linears[j].weight.grad = y_ratio*beta*v_critic.linears[j].weight.grad
                    v_critic.linears[j].bias.grad = y_ratio*beta*v_critic.linears[j].bias.grad
                optimizerV_C.step()
                optimizerV_C.zero_grad()

                v_value = v_critic(state)
                next_v_value = v_critic(next_state)
                delta = (reward + gamma * next_v_value - v_value).detach()
                error = delta - a_value
                a_critic_loss = error.pow(2)
                optimizerA_C.zero_grad()
                a_critic_loss.backward()
                w_ratio = w(state).detach()
                a_critic.linear1.weight.grad = w_ratio*beta*a_critic.linear1.weight.grad
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
    ratio_estimation_env.close()
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

    if load_W:
        path_W = config['w']['load_path']
        w = torch.load(path_W).to(device)
        print('W Model loaded')
    else:    
        w = W(state_size, w_h_layers_sizes, 1).to(device)
        torch.save(w, config['w']['initial_save_path'])

    if load_Y:
        path_Y = config['y']['load_path']
        y = torch.load(path_Y).to(device)
        print('Y Model loaded')
    else:    
        y = Y(state_size, y_h_layers_sizes, 1).to(device)
        torch.save(y, config['y']['initial_save_path'])

    if random_behv_prob < 1:
        behaviour_policy = torch.load(config['behaviour_policy_path']).to(device)
        print('Behaviour Policy loaded')
    else:
        behaviour_policy = None

    trainIters(actor, v_critic, a_critic, w, y, random_behv_prob, behaviour_policy, n_iters=iterations)
