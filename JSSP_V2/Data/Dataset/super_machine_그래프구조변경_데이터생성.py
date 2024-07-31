import os
import pandas as pd
from RLDataset_FJSSP import RLDataset_FJSSP
from JobShopEnv_FJSSP_machine_그래프_copy import JobShopEnv_FJSSP
import matplotlib.pyplot as plt
import logging
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import gc
import time
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

logging.basicConfig(filename='training_progress.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def swish(x):
    return x * torch.sigmoid(x)

def mish(x):
    return x * torch.tanh(F.softplus(x))

class MultiAgentGraph:
    def __init__(self, machines, jobs):
        self.machines = machines
        self.jobs = jobs
        self.graph = self.build_graph()

    def build_graph(self):
        graph = {}
        for machine in self.machines:
            graph[f'M_{machine}'] = {
                'type': 'machine',
                'static_edges': [],
                'dynamic_edges': {'executing': [], 'waiting': [], 'routing': []}
            }
        for job in self.jobs:
            graph[f'J_{job}'] = {
                'type': 'job',
                'dynamic_edges': {'executing': [], 'waiting': [], 'routing': []}
            }
        return graph

    def update_graph(self, state, op=None, machine=None):
        for node in self.graph.values():
            if node['type'] == 'machine':
                node['dynamic_edges'] = {'executing': [], 'waiting': [], 'routing': []}
            elif node['type'] == 'job':
                node['dynamic_edges'] = {'executing': [], 'waiting': [], 'routing': []}
                node['last_executed_machine'] = -1
                node['next_possible_machines'] = set()

        for job_id, job_info in state['jobs'].items():
            job_key = job_id
            current_machine = job_info['current_machine']
            if current_machine is not None:
                machine_key = f'M_{current_machine}'
                self.graph[machine_key]['dynamic_edges']['executing'].append(job_key)
                self.graph[job_key]['dynamic_edges']['executing'].append(machine_key)
                self.graph[job_key]['last_executed_machine'] = current_machine
            for machine_id in job_info['waiting_machines']:
                machine_key = f'M_{machine_id}'
                self.graph[machine_key]['dynamic_edges']['waiting'].append(job_key)
                self.graph[job_key]['dynamic_edges']['waiting'].append(machine_key)
            for machine_id in job_info['routing_machines']:
                machine_key = f'M_{machine_id}'
                self.graph[machine_key]['dynamic_edges']['routing'].append(job_key)
                self.graph[job_key]['dynamic_edges']['routing'].append(machine_key)
                self.graph[job_key]['next_possible_machines'].add(machine_id)

        # # job에 대한 디버그 출력 추가
        # for job_key, job_node in self.graph.items():
        #     if job_node['type'] == 'job':
        #         print(f"Debug - {job_key}: last_executed={job_node['last_executed_machine']}, "
        #             f"waiting={job_node['dynamic_edges']['waiting']}, "
        #             f"next_possible={job_node['next_possible_machines']}")

        # # 기계에 대한 디버그 출력 추가
        # for machine_key, machine_node in self.graph.items():
        #     if machine_node['type'] == 'machine':
        #         print(f"Debug - {machine_key}: executing={machine_node['dynamic_edges']['executing']}, "
        #             f"waiting={machine_node['dynamic_edges']['waiting']}, "
        #             f"routing={machine_node['dynamic_edges']['routing']}")

    def get_machine_features(self, machine_id):
        machine_node = self.graph[machine_id]
        n_jobs = len(self.jobs)
        
        # 현재 실행 중인 작업 번호 (없으면 -1)
        executing_job = -1
        for j in range(n_jobs):
            if f'J_{j}' in machine_node['dynamic_edges']['executing']:
                executing_job = j
                break
        
        # 대기 중인 job 갯수
        waiting_jobs_count = len(machine_node['dynamic_edges']['waiting'])
        
        # 향후 실행 가능한 job 갯수
        routing_jobs_count = len(machine_node['dynamic_edges']['routing'])
        
        features = [executing_job + 1, waiting_jobs_count, routing_jobs_count]
        
        # print(f"Features for {machine_id}: {features}")  # 디버깅 출력 추가
        return features

    def get_job_features(self, job_id):
        job_node = self.graph[job_id]
        
        last_executed_machine = job_node.get('last_executed_machine', -1) + 1
        
        waiting_machines = [int(m.split('_')[1]) for m in job_node['dynamic_edges']['waiting']]
        current_waiting_machine = min(waiting_machines) + 1 if waiting_machines else 0
        
        next_possible_machines_count = len(job_node.get('next_possible_machines', set()))
        
        features = [last_executed_machine, current_waiting_machine, next_possible_machines_count]
        
        # print(f"Features for {job_id}: {features}")  # 디버깅 출력 추가
        return features

    def get_state_features(self, agent_id):
        if self.graph[agent_id]['type'] == 'machine':
            features = self.get_machine_features(agent_id)
        elif self.graph[agent_id]['type'] == 'job':
            features = self.get_job_features(agent_id)
        
        # print(f"Features for {agent_id}: {features}")  # 디버깅 출력
        return features
        
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
        # print(f"SumTree initialized with capacity: {capacity}")  # 디버깅

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
        # print(f"Data added to SumTree. Total entries: {self.n_entries}")  # 디버깅

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

# DuelingDQN for machine agents
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, action_size, dropout_rate=0.2):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, action_size)
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
        print(f"DuelingDQN initialized with input_dim: {input_dim}, action_size: {action_size}")


    def forward(self, x):
        feature = self.feature(x)
        advantage = self.advantage(feature)
        value = self.value(feature)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

# DDQNAgent for machine agents
class DDQNAgent:
    def __init__(self, input_dim, action_size, update_target_frequency=100, replay_start_size=1600, batch_size=16):
        self.input_dim = input_dim
        self.action_size = action_size
        self.memory = SumTree(10000)
        self.gamma = 0
        self.epsilon = 0.99
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0001
        self.model = DuelingDQN(input_dim, action_size)
        self.target_model = DuelingDQN(input_dim, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss(reduction='none')
        self.update_target_frequency = update_target_frequency
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size
        self.update_counter = 0
        self.episode_count = 0
        self.PER_e = 0.01
        self.PER_a = 0.6
        self.PER_b = 0.4
        self.PER_b_increment_per_sampling = 0.001
        self.absolute_error_upper = 1.
        self.temperature = 0.5  # 클래스 초기화 시 설정
        
        print(f"DDQNAgent initialized with input_dim: {input_dim}, action_size: {action_size}")  # 디버깅

    def remember(self, state, action, reward, next_state, done):
        # state와 next_state 처리
        if isinstance(state, dict):
            state = np.array(list(state.values())).flatten()
        elif not isinstance(state, np.ndarray):
            state = np.array(state)
        
        if isinstance(next_state, dict):
            next_state = np.array(list(next_state.values())).flatten()
        elif not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state)
        
        # 상태가 1차원이 아닌 경우 flatten
        if state.ndim > 1:
            state = state.flatten()
        if next_state.ndim > 1:
            next_state = next_state.flatten()
        
        # action 처리
        if isinstance(action, dict):
            action = next(iter(action.values()))
        elif isinstance(action, (list, tuple)):
            action = action[0]
        
        # action 값을 클리핑
        action = max(0, min(action, self.action_size - 1))
        
        experience = (state, action, reward, next_state, done)
        max_priority = np.max(self.memory.tree[-self.memory.capacity:])
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        self.memory.add(max_priority, experience)
        # print(f"Experience added to memory. Memory size: {self.memory.n_entries}")  # 디버깅


    def act(self, state, valid_operations):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_operations)

        if isinstance(state, dict):
            state = np.concatenate([np.array(v, dtype=np.float32).flatten() for v in state.values()])
        elif isinstance(state, (list, tuple)):
            state = np.concatenate([np.array(v, dtype=np.float32).flatten() for v in state])
        else:
            state = np.array(state, dtype=np.float32).flatten()

        state = torch.FloatTensor(state).unsqueeze(0)

        q_values = self.model(state).squeeze(0)
        valid_q_indices = [i for i, op in enumerate(valid_operations) if i < self.action_size]
        best_action_index = max(valid_q_indices, key=lambda i: q_values[i].item())

        selected_action = valid_operations[best_action_index]
        # print(f"Action selected: {selected_action}")  # 디버깅
        return selected_action

    def update_epsilon(self, global_episode_count):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        print(f"Epsilon updated: {self.epsilon}")  # 디버깅

    def replay(self):
        if self.memory.n_entries < self.replay_start_size:
            return

        minibatch = []
        idxs = []
        segment = self.memory.total() / self.batch_size
        priorities = []

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.memory.get(s)
            priorities.append(p)
            minibatch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.memory.total()
        is_weight = np.power(self.memory.n_entries * sampling_probabilities, -self.PER_b)
        is_weight /= is_weight.max()

        # 상태의 최대 길이 찾기
        max_state_length = self.input_dim  # 입력 차원에 맞게 패딩
        
        # 패딩된 상태 생성
        states = np.array([np.pad(each[0], (0, max_state_length - len(each[0])), 'constant') for each in minibatch])
        
        # 액션 처리 (이미 정규화되어 있음)
        actions = np.array([each[1] for each in minibatch])
        
        rewards = np.array([each[2] for each in minibatch])
        next_states = np.array([np.pad(each[3], (0, max_state_length - len(each[3])), 'constant') for each in minibatch])
        dones = np.array([each[4] for each in minibatch])

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        is_weight = torch.FloatTensor(is_weight)

        # 디버깅 출력 추가
        # print(f"States shape: {states.shape}")
        # print(f"Actions shape: {actions.shape}")
        # print(f"Rewards shape: {rewards.shape}")
        # print(f"Next states shape: {next_states.shape}")
        # print(f"Dones shape: {dones.shape}")
        # print(f"Is weight shape: {is_weight.shape}")

        actions = torch.clamp(actions, 0, self.action_size - 1)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states)
        next_q_state_values = self.target_model(next_states)

        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)

        loss = self.criterion(q_values, expected_q_value.detach()) * is_weight
        prios = (loss + self.PER_e) ** self.PER_a
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, prios[i].item())

        self.update_counter += 1
        if self.update_counter % self.update_target_frequency == 0:
            self.update_target_model()

        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])
        print(f"Replay performed. Loss: {loss.item()}")  # 디버깅

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        print("Target model updated")  # 디버깅

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        # print(f"Model loaded from {name}")  # 디버깅

    def save(self, name):
        torch.save(self.model.state_dict(), name)
        # print(f"Model saved to {name}")  # 디버깅

    def set_train_mode(self):
        self.model.train()
        self.target_model.train()
        print("Model set to training mode")

    def set_eval_mode(self):
        self.model.eval()
        self.target_model.eval()
        print("Model set to training mode")

# SumTree_super for supervisor agent
class SumTree_super:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
        # print(f"SumTree_super initialized with capacity: {capacity}")  # 디버깅

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
        # print(f"Data added to SumTree_super. Total entries: {self.n_entries}")  # 디버깅

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

# DuelingDQN_super for supervisor agent
class DuelingDQN_super(nn.Module):
    def __init__(self, input_dim, action_size, dropout_rate=0.2):
        super(DuelingDQN_super, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, action_size)
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
        # print(f"DuelingDQN_super initialized with input_dim: {input_dim}, action_size: {action_size}")  # 디버깅

    def forward(self, x):
        feature = self.feature(x)
        advantage = self.advantage(feature)
        value = self.value(feature)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

# DDQNAgent_super for supervisor agent
class DDQNAgent_super:
    def __init__(self, input_dim, action_size, update_target_frequency=100, replay_start_size=8000, batch_size=16):
        self.input_dim = input_dim
        self.action_size = action_size
        self.memory = SumTree_super(10000)
        self.gamma = 0.9
        self.epsilon = 0.99
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0001
        self.model = DuelingDQN_super(input_dim, action_size)
        self.target_model = DuelingDQN_super(input_dim, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss(reduction='none')
        self.update_target_frequency = update_target_frequency
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size
        self.update_counter = 0
        self.episode_count = 0
        self.PER_e = 0.01
        self.PER_a = 0.6
        self.PER_b = 0.4
        self.PER_b_increment_per_sampling = 0.001
        self.absolute_error_upper = 1.
        self.used_machines = set()
        self.temperature = 0.5  # 클래스 초기화 시 설정
        print(f"DDQNAgent_super initialized with input_dim: {input_dim}, action_size: {action_size}")  # 디버깅

    def remember(self, state, action, reward, next_state, done):
        if isinstance(state, dict):
            state = np.concatenate(list(state.values()))
        if isinstance(next_state, dict):
            next_state = np.concatenate(list(next_state.values()))
        
        experience = (state, action, reward, next_state, done)
        max_priority = np.max(self.memory.tree[-self.memory.capacity:])
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        self.memory.add(max_priority, experience)
        # print(f"Experience added to memory. Memory size: {self.memory.n_entries}")  # 디버깅

    def select_machine(self, available_machines, state_features):
        if np.random.rand() <= self.epsilon:
            return random.choice(available_machines)

        state = torch.FloatTensor(state_features).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state).squeeze(0)

        valid_q_values = {machine: q_values[machine].item() for machine in available_machines}
        selected_machine = max(valid_q_values, key=valid_q_values.get)

        # print(f"Machine {selected_machine} selected")  # 디버깅
        return selected_machine
            
    def update(self, machine_id):
        self.used_machines.add(machine_id)
        # print(f"Machine {machine_id} marked as used")  # 디버깅

    def is_machine_used(self, machine_id):
        return machine_id in self.used_machines

    def reset(self):
        self.used_machines.clear()
        # print("Used machines reset")  # 디버깅

    def update_epsilon(self, global_episode_count):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        print(f"Epsilon updated: {self.epsilon}")  # 디버깅

    def replay(self):
        if self.memory.n_entries < self.replay_start_size:
            return

        minibatch = []
        idxs = []
        segment = self.memory.total() / self.batch_size
        priorities = []

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.memory.get(s)
            priorities.append(p)
            minibatch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.memory.total()
        is_weight = np.power(self.memory.n_entries * sampling_probabilities, -self.PER_b)
        is_weight /= is_weight.max()

        # 상태의 최대 길이 찾기
        max_state_length = max(len(each[0]) for each in minibatch)

        # 패딩된 상태 생성
        states = np.array([np.pad(each[0], (0, max_state_length - len(each[0])), 'constant') for each in minibatch])
        
        # 액션 처리 (실제 선택된 액션 사용)
        actions = []
        for each in minibatch:
            if isinstance(each[1], dict):
                actions.append(next(iter(each[1].values())))
            else:
                actions.append(each[1])
        actions = np.array(actions)
        
        rewards = np.array([each[2] for each in minibatch])
        next_states = np.array([np.pad(each[3], (0, max_state_length - len(each[3])), 'constant') for each in minibatch])
        dones = np.array([each[4] for each in minibatch])

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        is_weight = torch.FloatTensor(is_weight)

        # 디버깅 출력 추가
        # print(f"States shape super: {states.shape}")
        # print(f"Actions shape super: {actions.shape}")
        # print(f"Rewards shape super: {rewards.shape}")
        # print(f"Next states shape super: {next_states.shape}")
        # print(f"Dones shape super: {dones.shape}")
        # print(f"Is weight shape super: {is_weight.shape}")

        actions = torch.clamp(actions, 0, self.action_size - 1)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states)
        next_q_state_values = self.target_model(next_states)

        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)

        loss = self.criterion(q_values, expected_q_value.detach()) * is_weight
        prios = (loss + self.PER_e) ** self.PER_a
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, prios[i].item())

        self.update_counter += 1
        if self.update_counter % self.update_target_frequency == 0:
            self.update_target_model()

        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])
        print(f"Replay performed_super. Loss: {loss.item()}")  # 디버깅

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        print("Target model updated_super")  # 디버깅

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        # print(f"Model loaded from {name}")  # 디버깅

    def save(self, name):
        torch.save(self.model.state_dict(), name)
        # print(f"Model saved to {name}")  # 디버깅

    def set_train_mode(self):
        self.model.train()
        self.target_model.train()
        print("Supervisor model set to training mode")

    def set_eval_mode(self):
        self.model.eval()
        self.target_model.eval()
        print("Supervisor model set to evaluation mode")

class MultiAgentSystem:
    def __init__(self, machines, jobs, state_size, action_size):
        self.graph = MultiAgentGraph(machines, jobs)
        self.machines = machines
        self.jobs = jobs
        self.action_size = action_size
        self.agents = {}
        self.supervisor = None
        # self.replay_start_size = 1000
        self.global_episode_count = 0
        self.actual_actions = {}
        print(f"MultiAgentSystem initialized with {len(machines)} machines and {len(jobs)} jobs")  # 디버깅

    def initialize_agents(self, sample_state):
        input_dim = self.calculate_input_dim(sample_state)
        self.agents = {f'M_{machine}': DDQNAgent(input_dim, self.action_size) for machine in self.machines}
        job_input_dim = self.calculate_job_input_dim(sample_state)
        self.agents.update({f'J_{job}': DDQNAgent(input_dim, self.action_size) for job in self.jobs})
        supervisor_input_dim = self.calculate_supervisor_input_dim(sample_state)
        self.supervisor = DDQNAgent_super(supervisor_input_dim, len(self.machines))
        print(f"Agents initialized. Number of machine agents: {len(self.agents)}")  # 디버깅

    def calculate_input_dim(self, state):
        machine_features_dim = len(self.graph.get_state_features(f'M_0'))  # get_machine_features를 get_state_features로 변경
        env_state_dim = (
            1 +  # current_time
            len(state['job_completion']) +
            len(state['machine_available_time']) +
            len(state['machine_utilization']) +
            len(state['job_queue_length']) +
            len(state['job_progress']) +
            len(state['remaining_job_time']) +
            sum(len(status) for status in state['job_op_status']) +
            len(state['machine_status'])
        )
        total_dim = machine_features_dim + env_state_dim
        print(f"Calculated input dimension for machine agents: {total_dim}")  # 디버깅 출력 추가
        return total_dim

    def calculate_job_input_dim(self, state):
        machine_features_dim = len(self.graph.get_state_features(f'J_0'))  # get_machine_features를 get_state_features로 변경
        env_state_dim = (
            1 +  # current_time
            len(state['job_completion']) +
            len(state['machine_available_time']) +
            len(state['machine_utilization']) +
            len(state['job_queue_length']) +
            len(state['job_progress']) +
            len(state['remaining_job_time']) +
            sum(len(status) for status in state['job_op_status']) +
            len(state['machine_status'])
        )
        total_dim = machine_features_dim + env_state_dim
        print(f"Calculated input dimension for machine agents: {total_dim}")  # 디버깅 출력 추가
        return total_dim

    def calculate_supervisor_input_dim(self, state):
        machine_features = sum(len(self.graph.get_state_features(f'M_{i}')) for i in self.machines)
        job_features = sum(len(self.graph.get_state_features(f'J_{j}')) for j in self.jobs)
        env_state_length = (
            1 +  # current_time
            len(state['job_completion']) +
            len(state['machine_available_time']) +
            len(state['machine_utilization']) +
            len(state['job_queue_length']) +
            len(state['job_progress']) +
            len(state['remaining_job_time']) +
            sum(len(status) for status in state['job_op_status']) +
            len(state['machine_status'])
        )
        total_dim = machine_features + job_features + env_state_length
        # total_dim = job_features + env_state_length
        return total_dim

    def reset(self, reset_epsilon=True):
        if reset_epsilon:
            for agent in self.agents.values():
                agent.epsilon = 0.999  # 에이전트의 탐험률 초기화 (필요에 따라 조정)
            self.supervisor.epsilon = 0.999  # SupervisorAgent의 탐험률 초기화
        self.supervisor.reset()  # SupervisorAgent 초기화
        self.set_train_mode()  # 학습 모드로 설정

    def reset_predict(self, reset_epsilon=True):
        for agent in self.agents.values():
            if reset_epsilon:
                agent.epsilon = 0.01  # 에이전트의 탐험률 초기화 (필요에 따라 조정)
        if reset_epsilon:
            self.supervisor.epsilon = 0.01 # SupervisorAgent의 탐험률 초기화
        self.supervisor.reset()  # SupervisorAgent 초기화
        self.set_eval_mode()  # 평가 모드로 설정


    def act(self, state, valid_actions):
        self.graph.update_graph(state)
        
        supervisor_features = self.get_supervisor_features(state)
        available_machines = [m for job, op, m in valid_actions if not self.supervisor.is_machine_used(m)]
        
        if not available_machines:
            self.supervisor.reset()
            available_machines = [m for job, op, m in valid_actions]

        if not available_machines:
            print("No available machines")  # 디버깅
            return None, None

        selected_machine = self.supervisor.select_machine(available_machines, supervisor_features)

        machine_features = self.graph.get_machine_features(f'M_{selected_machine}')
        env_state = np.concatenate([
            np.array([state['current_time']]),
            state['job_completion'],
            state['machine_available_time'],
            state['machine_utilization'],
            state['job_queue_length'],
            state['job_progress'],
            state['remaining_job_time'],
            np.array([status for job_status in state['job_op_status'] for status in job_status]),
            np.array(state['machine_status'], dtype=int)
        ])
        # print(f"Machine features shape: {np.array(machine_features).shape}")  # 디버깅 출력
        # print(f"Env state shape: {env_state.shape}")  # 디버깅 출력
        
        machine_state = np.concatenate([
            np.array(machine_features, dtype=np.float32),
            env_state.astype(np.float32)
        ])
        
        # print(f"Machine state shape: {machine_state.shape}")  # 디버깅 출력
        
        valid_operations = [(job, op) for job, op, machine in valid_actions if machine == selected_machine]
        
        if not valid_operations:
            print(f"No valid operations for machine {selected_machine}")  # 디버깅
            return None, None
        
        selected_operation = self.agents[f'M_{selected_machine}'].act(machine_state, valid_operations)
        
        self.supervisor.update(selected_machine)
        # print(f"Selected machine: {selected_machine}, Selected operation: {selected_operation}")  # 디버깅
        return selected_machine, selected_operation
            
    def get_supervisor_features(self, state):
        machine_features = [feature for m in self.machines for feature in self.graph.get_state_features(f'M_{m}')]
        job_features = [feature for j in self.jobs for feature in self.graph.get_state_features(f'J_{j}')]
        env_features = [
            state['current_time'],
            *state['job_completion'],
            *state['machine_available_time'],
            *state['machine_utilization'],
            *state['job_queue_length'],
            *state['job_progress'],
            *state['remaining_job_time'],
            *[status for job_status in state['job_op_status'] for status in job_status],
            *state['machine_status']
        ]
        features = machine_features + job_features + env_features
        # features = job_features + env_features
       
        return features

    def remember(self, agent_id, state, action, reward, next_state, done):
        if agent_id == 'supervisor':
            self.supervisor.remember(state, action, reward, next_state, done)
        else:
            if isinstance(action, dict):
                action = next(iter(action.values()))
            self.agents[agent_id].remember(state, action, reward, next_state, done)
        # print(f"Experience remembered for agent {agent_id}")  # 디버깅

    def replay(self):
        for agent_id, agent in self.agents.items():
            if agent.memory.n_entries >= agent.replay_start_size:
                agent.replay()
        if self.supervisor.memory.n_entries >= self.supervisor.replay_start_size:
            self.supervisor.replay()
        # print("Replay performed for all eligible agents and supervisor")

    def update_target_models(self):
        for agent in self.agents.values():
            agent.update_target_model()
        self.supervisor.update_target_model()
        # print("Target models updated for all agents and supervisor")  # 디버깅

    def load(self, load_paths):
        for agent_id, agent in self.agents.items():
            if agent_id in load_paths:
                agent.load(load_paths[agent_id])
                print(f"Model loaded from {load_paths[agent_id]}")
        self.supervisor.load(load_paths['supervisor'])
        # print(f"Model loaded from {load_paths['supervisor']}")
        # print("Models loaded for all agents and supervisor")
        
    def save(self, save_paths):
        for agent_id, agent in self.agents.items():
            if agent_id in save_paths:
                agent.save(save_paths[agent_id])
                print(f"Model saved to {save_paths[agent_id]}")
        self.supervisor.save(save_paths['supervisor'])
        # print(f"Model saved to {save_paths['supervisor']}")
        # print("Models saved for all agents and supervisor")

    def memory_size(self):
        total_size = sum(agent.memory.n_entries for agent in self.agents.values()) + self.supervisor.memory.n_entries
        # print(f"Total memory size: {total_size}")  # 디버깅
        return total_size

    def set_train_mode(self):
        for agent in self.agents.values():
            agent.set_train_mode()
        self.supervisor.set_train_mode()
        print("supervisor set to training mode")

    def set_eval_mode(self):
        for agent in self.agents.values():
            agent.set_eval_mode()
        self.supervisor.set_eval_mode()
        print("supervisor set to evaluation mode")

    def end_episode(self, state, done, is_training=True):
        if is_training:
            self.global_episode_count += 1
            
            state_features = {}
            for agent_id in self.agents:
                if agent_id.startswith('M_'):
                    machine_features = self.graph.get_machine_features(agent_id)
                    machine_features = np.array(list(machine_features))

                    state_features[agent_id] = np.concatenate((
                        machine_features,
                        np.array([state['current_time']]),
                        np.array(state['job_completion']),
                        np.array(state['machine_available_time']),
                        np.array(state['machine_utilization']),
                        np.array(state['job_queue_length']),
                        np.array(state['job_progress']),
                        np.array(state['remaining_job_time']),
                        np.array([status for job_status in state['job_op_status'] for status in job_status]),
                        np.array(state['machine_status'], dtype=int)
                    ))
                elif agent_id.startswith('J_'):
                    job_features = self.graph.get_job_features(agent_id)
                    job_features = np.array(list(job_features))

                    state_features[agent_id] = np.concatenate((
                        job_features,
                        np.array([state['current_time']]),
                        np.array(state['job_completion']),
                        np.array(state['machine_available_time']),
                        np.array(state['machine_utilization']),
                        np.array(state['job_queue_length']),
                        np.array(state['job_progress']),
                        np.array(state['remaining_job_time']),
                        np.array([status for job_status in state['job_op_status'] for status in job_status]),
                        np.array(state['machine_status'], dtype=int)
                    ))
            supervisor_features = self.get_supervisor_features(state)

            # print(supervisor_features)
            
            for agent_id, agent in self.agents.items():
                actual_action = self.actual_actions.get(agent_id, 0)
                if isinstance(actual_action, dict):
                    actual_action = next(iter(actual_action.values()))
                # agent.remember(state_features[agent_id], actual_action, reward_machine, state_features[agent_id], done)
                agent.update_epsilon(self.global_episode_count)
            
            supervisor_action = self.actual_actions.get('supervisor', 0)
            if isinstance(supervisor_action, dict):
                supervisor_action = next(iter(supervisor_action.values()))
            # self.supervisor.remember(supervisor_features, supervisor_action, reward_machine, supervisor_features, done)
            self.supervisor.update_epsilon(self.global_episode_count)

            for agent in self.agents.values():
                if agent.memory.n_entries >= agent.replay_start_size:
                    agent.replay()
            
            if self.supervisor.memory.n_entries >= self.supervisor.replay_start_size:
                self.supervisor.replay()
            
        print(f"Episode {self.global_episode_count} ended.")  # 디버깅
        # print(f"Episode {self.global_episode_count} ended. Reward task: {reward_task}, Reward machine: {reward_machine}")  # 디버깅


def train_individual_models(datasets, num_episodes_per_dataset):
    # 초기 데이터셋으로 multi_agent_system 초기화
    data = RLDataset_FJSSP(datasets[0])
    action_size = data.n_job

    multi_agent_system = MultiAgentSystem(machines=range(data.n_machine), jobs=range(data.n_job), state_size=None, action_size=action_size)
    
    for i, dataset in enumerate(datasets):
        print(f"Starting training for dataset {dataset} ({i+1}/{len(datasets)})")

        data = RLDataset_FJSSP(dataset)
        
        process_times = [[(m, t) for m, t in ops] for job in data.op_data for ops in job]
        machine_sequence = [[[(m, t) for m, t in ops] for ops in job] for job in data.op_data]

        env = JobShopEnv_FJSSP(process_times, machine_sequence, solutions=None)
        initial_state = env.reset()

        # 모델을 초기화하지 않고, 그대로 사용
        if i == 0:
            multi_agent_system.initialize_agents(initial_state)
        else:
            # 이전 모델을 로드
            # 모델을 최종적으로 저장
            agent_save_paths = {}
            for agent_id in multi_agent_system.agents.keys():
                if agent_id.startswith('M_'):
                    agent_save_paths[agent_id] = f"{agent_id}.pth"
                elif agent_id.startswith('J_'):
                    agent_save_paths[agent_id] = f"{agent_id}.pth"
            agent_save_paths['supervisor'] = "supervisor.pth"

            multi_agent_system.save(agent_save_paths)
            multi_agent_system.load(agent_save_paths)

        # 데이터셋 변경 시 epsilon 값 유지 여부 확인
        print(f"Supervisor epsilon value before reset: {multi_agent_system.supervisor.epsilon}")
        multi_agent_system.reset(reset_epsilon=False)
        print(f"Supervisor epsilon value after reset: {multi_agent_system.supervisor.epsilon}")

        # 학습 모드로 설정
        multi_agent_system.set_train_mode()  # 학습 시작 전 한 번만 호출


        for episode in range(num_episodes_per_dataset):
            state = env.reset()
            multi_agent_system.supervisor.reset()  # SupervisorAgent 초기화
            done = False
            episode_reward = 0
            step_count = 0

            while not done:
                state = env.get_state()
                multi_agent_system.graph.update_graph(state)

                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    print("No valid actions available. Ending episode.")
                    break

                selected_machine, selected_operation = multi_agent_system.act(state, valid_actions)

                if selected_machine is None or selected_operation is None:
                    print("No valid selection made. Ending episode.")
                    break

                selected_job, selected_op = selected_operation
                next_state, done, step_reward = env.step(selected_job, selected_op, selected_machine)

                multi_agent_system.graph.update_graph(next_state)

                # 각 에이전트의 state features 업데이트
                machine_features = multi_agent_system.graph.get_machine_features(f'M_{selected_machine}')
                supervisor_features = multi_agent_system.get_supervisor_features(state)

                # 실제 선택된 액션 저장
                multi_agent_system.remember(f'M_{selected_machine}', 
                                            machine_features, 
                                            selected_operation,  # 실제 선택된 operation
                                            step_reward,
                                            multi_agent_system.graph.get_machine_features(f'M_{selected_machine}'), 
                                            done)

                multi_agent_system.remember('supervisor', 
                                            supervisor_features, 
                                            selected_machine,  # 실제 선택된 machine
                                            step_reward, 
                                            multi_agent_system.get_supervisor_features(next_state), 
                                            done)

                state = next_state
                episode_reward += step_reward
                step_count += 1

                if step_count % 50 == 0:
                    gc.collect()
                    print(f"Step {step_count} completed. Memory cleaned.")

                if done:
                    reward_task, reward_machine = env.calculate_episode_rewards()
                    # reward_task += episode_reward
                    # reward_machine += episode_reward
                    print(f"Training: Episode {episode+1} processed. Reward Task: {reward_task}, Reward Machine: {reward_machine}, episode reward every step: {episode_reward}")

            multi_agent_system.end_episode(state, done, is_training=True)
            
            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes_per_dataset} completed for dataset {dataset}")
                gc.collect()

        print(f"Completed training for dataset {dataset} ({i+1}/{len(datasets)})")
        gc.collect()

    agent_save_paths = {}
    for agent_id in multi_agent_system.agents.keys():
        if agent_id.startswith('M_'):
            agent_save_paths[agent_id] = f"{agent_id}.pth"
        elif agent_id.startswith('J_'):
            agent_save_paths[agent_id] = f"{agent_id}.pth"
    agent_save_paths['supervisor'] = "supervisor.pth"

    multi_agent_system.save(agent_save_paths)

    return multi_agent_system, None, data.n_machine


def predict(multi_agent_system, env, test_dataset, num_predictions=1, max_steps=100000000):
    valid_solution_count = 0
    all_predictions = []
    step_count = 0

    # 모델을 로드
    agent_load_paths = {}
    for agent_id in multi_agent_system.agents.keys():
        if agent_id.startswith('M_'):
            agent_load_paths[agent_id] = f"{agent_id}.pth"
        elif agent_id.startswith('J_'):
            agent_load_paths[agent_id] = f"{agent_id}.pth"
    agent_load_paths['supervisor'] = "supervisor.pth"

    multi_agent_system.load(agent_load_paths)
    multi_agent_system.set_eval_mode()  # 예측 시작 전 호출

    while valid_solution_count < num_predictions and step_count < max_steps:
        state = env.reset()
        solution = []
        done = False
        episode_reward = 0
        multi_agent_system.reset(reset_epsilon=False)

        while not done and step_count < max_steps:
            state = env.get_state()
            multi_agent_system.graph.update_graph(state)

            valid_actions = env.get_valid_actions()
            if not valid_actions:
                print("No valid actions available. Ending episode.")
                break

            selected_machine, selected_operation = multi_agent_system.act(state, valid_actions)

            if selected_machine is None or selected_operation is None:
                print("No valid selection made. Ending episode.")
                break

            selected_job, selected_op = selected_operation
            next_state, done, step_reward = env.step(selected_job, selected_op, selected_machine)

            multi_agent_system.graph.update_graph(next_state)

            duration = next(t for m, t in env.machine_sequence[selected_job][selected_op] if m == selected_machine)
            solution.append((selected_job, selected_op, selected_machine, duration))
            state = next_state
            episode_reward += step_reward
            step_count += 1

            if step_count % 50 == 0:
                gc.collect()

            print(f"Step reward: {step_reward}")

        if done:
            reward_task, reward_machine = env.calculate_episode_rewards()
            # reward_task += episode_reward
            # reward_machine += episode_reward
            print(f"Prediction: Episode {valid_solution_count+1}/{num_predictions} processed. "
                  f"Reward Task: {reward_task}, Reward Machine: {reward_machine}, Episode every step Reward: {episode_reward}")
            valid_solution_count += 1
            all_predictions.append(solution)
            gc.collect()

    multi_agent_system.end_episode(state, done, is_training=False)

    if valid_solution_count != num_predictions:
        print(f"Warning: Only {valid_solution_count} valid solutions were collected.")

    return all_predictions


def main():
    datasets = [
        # 'fjsspdataset/HurinkEdata7.fjs',
        'fjssprandom_10_5/dataset_1.fjs',
        'fjssprandom_10_5/dataset_2.fjs',
        'fjssprandom_10_5/dataset_3.fjs',
        'fjssprandom_10_5/dataset_4.fjs', 
        'fjssprandom_10_5/dataset_5.fjs',
        'fjssprandom_10_5/dataset_6.fjs',
        'fjssprandom_10_5/dataset_7.fjs',
        'fjssprandom_10_5/dataset_8.fjs', 
        'fjssprandom_10_5/dataset_9.fjs',
        'fjssprandom_10_5/dataset_10.fjs',
        'fjssprandom_10_5/dataset_11.fjs',
        'fjssprandom_10_5/dataset_12.fjs', 
        'fjssprandom_10_5/dataset_13.fjs',
        'fjssprandom_10_5/dataset_14.fjs',
        'fjssprandom_10_5/dataset_15.fjs',
        'fjssprandom_10_5/dataset_16.fjs', 
        'fjssprandom_10_5/dataset_17.fjs',
        'fjssprandom_10_5/dataset_18.fjs',
        'fjssprandom_10_5/dataset_19.fjs',
        'fjssprandom_10_5/dataset_20.fjs', 
    ]
    num_episodes_per_dataset = 50
#abz5
    multi_agent_system, max_state_size, action_size = train_individual_models(datasets, num_episodes_per_dataset)

    test_dataset = RLDataset_FJSSP('fjsspdataset/HurinkEdata7.fjs')

    logging.info("Training completed for individual models.")
    print("Training completed for individual models.")

    logging.info("Creating environment object.")
    print("Creating environment object.")

    process_times = [[(m, t) for m, t in ops] for job in test_dataset.op_data for ops in job]
    machine_sequence = [[[(m, t) for m, t in ops] for ops in job] for job in test_dataset.op_data]

    env = JobShopEnv_FJSSP(process_times, machine_sequence, solutions=None)

    logging.info("Starting prediction process.")
    print("Starting prediction process.")
    all_predictions = predict(multi_agent_system, env, test_dataset, num_predictions=300, max_steps=100000000)

    logging.info(f"Predicted Solutions: {all_predictions}")
    print("Predicted Solutions:", all_predictions)

    best_solution = None
    best_makespan = float('inf')
    for solution in all_predictions:
        makespan = calculate_makespan(solution, env)
        if makespan < best_makespan:
            best_makespan = makespan
            best_solution = solution

    logging.info(f"Best Solution: {best_solution}")
    print("Best Solution:", best_solution)

    ga_initial_population = [job * env.n_machines + op for job, op, _, _ in best_solution]
    logging.info(f"GA Initial Population: {ga_initial_population}")
    print("GA Initial Population:", ga_initial_population)

    draw_gantt_chart(best_solution, env)

def calculate_makespan(solution, env):
    job_start_times = {job: 0 for job in range(env.n_jobs)}
    machine_avail_times = {machine: 0 for machine in range(env.n_machines)}
    makespan = 0

    for job, op, machine, duration in solution:
        start_time = max(job_start_times[job], machine_avail_times[machine])
        end_time = start_time + duration
        job_start_times[job] = end_time
        machine_avail_times[machine] = end_time
        makespan = max(makespan, end_time)

    return makespan

def generate_colors(n):
    colors = plt.colormaps['tab20'](range(n))
    return [mcolors.rgb2hex(c) for c in colors]

def color(row, color_map):
    return color_map[row['Job']]

def draw_gantt_chart(predictions, env):
    job_start_times = {job: 0 for job in range(env.n_jobs)}
    machine_avail_times = {machine: 0 for machine in range(env.n_machines)}
    makespan = 0

    gantt_chart = []
    job_info = []

    for job, op, machine, duration in predictions:
        start_time = max(job_start_times[job], machine_avail_times[machine])
        end_time = start_time + duration

        gantt_chart.append((machine, start_time, end_time, job))
        job_info.append((job, op, machine, start_time, end_time, duration))
        
        job_start_times[job] = end_time
        machine_avail_times[machine] = end_time
        makespan = max(makespan, end_time)

    gantt_df = pd.DataFrame(gantt_chart, columns=['Machine', 'Start', 'End', 'Job'])
    job_info_df = pd.DataFrame(job_info, columns=['Job', 'Operation', 'Machine', 'Start', 'End', 'Duration'])

    unique_jobs = gantt_df['Job'].unique()
    colors = plt.cm.get_cmap('tab20', len(unique_jobs)).colors
    color_map = {job: mcolors.rgb2hex(colors[i]) for i, job in enumerate(unique_jobs)}

    gantt_df['Color'] = gantt_df.apply(lambda row: color_map[row['Job']], axis=1)
    gantt_df['Delta'] = gantt_df['End'] - gantt_df['Start']

    fig, ax = plt.subplots(1, figsize=(16*0.8, 9*0.8))
    ax.barh(gantt_df['Machine'], gantt_df['Delta'], left=gantt_df['Start'], color=gantt_df['Color'], edgecolor='black')

    # 수정된 부분: 세로축을 1 간격으로 설정
    ax.set_yticks(np.arange(0, env.n_machines + 1, 1))
    ax.set_yticklabels(np.arange(0, env.n_machines + 1, 1))

    legend_elements = [Patch(facecolor=color_map[job], label=f'Job {job}') for job in unique_jobs]
    plt.legend(handles=legend_elements)
    plt.title('Gantt Chart', size=24)
    ax.set_xlim(0, makespan + 10)

    plt.text(makespan, -1, f'{makespan}', color='black', ha='center', va='center')
    plt.text(makespan, ax.get_ylim()[1], f'Max Makespan: {makespan}', color='red', ha='right', va='top')

    plt.xlabel('Time')
    plt.ylabel('Machine')

    plt.subplots_adjust(left=0.2, top=0.7)

    plt.show()


if __name__ == "__main__":
    main()
