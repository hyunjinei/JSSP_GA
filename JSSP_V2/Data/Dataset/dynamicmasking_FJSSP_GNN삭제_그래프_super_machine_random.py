import os
import pandas as pd
from RLDataset_FJSSP import RLDataset_FJSSP
from JobShopEnv_FJSSP_machine_그래프 import JobShopEnv_FJSSP
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
# from multi_agent_graph import MultiAgentGraph

logging.basicConfig(filename='training_progress.log', level=logging.INFO, format='%(asctime)s - %(message)s')


class DQN(nn.Module):
    def __init__(self, input_dim, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x


class DDQNAgent:
    def __init__(self, input_dim, action_size, update_target_frequency=80, replay_start_size=100, batch_size=80):
        self.input_dim = input_dim
        self.action_size = action_size
        self.memory = deque(maxlen=20000000)
        self.gamma = 0.5
        self.epsilon = 0.99
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99
        self.learning_rate = 0.00001
        self.model = DQN(input_dim, action_size)
        self.target_model = DQN(input_dim, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()
        self.update_target_frequency = update_target_frequency
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size
        self.update_counter = 0
        self.episode_count = 0

    def remember(self, state, action, reward, next_state, done):
        if isinstance(action, dict):
            action = next(iter(action.values()))
        elif isinstance(action, (list, tuple)):
            action = action[0]
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_operations):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_operations)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        
        q_values = self.model(state).squeeze(0)
        valid_q_indices = [i for i, op in enumerate(valid_operations) if i < self.action_size]
        best_action_index = max(valid_q_indices, key=lambda i: q_values[i].item())
        
        target_q_values = self.target_model(state).squeeze(0)
        best_action_value = target_q_values[best_action_index].item()
        
        return valid_operations[best_action_index]


    def update_epsilon(self, global_episode_count):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # if global_episode_count % 5 == 0:
        #     print(f'Episode {global_episode_count}, Epsilon: {self.epsilon:.4f}')

    def replay(self):
        if len(self.memory) < self.replay_start_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, done = zip(*minibatch)

        states = torch.FloatTensor(np.array(states))
        # print("Raw actions:", actions)
        actions = [min(max(action, 0), self.action_size - 1) if isinstance(action, (int, np.integer)) 
                   else min(max(next(iter(action.values())) if isinstance(action, dict) else action, 0), self.action_size - 1) 
                   for action in actions]
        # print("Processed actions:", actions)
        actions = torch.LongTensor(actions)
        # print("Action tensor:", actions)
        # print("Action tensor max:", actions.max())
        # print("Action tensor min:", actions.min())
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        done = torch.FloatTensor(done)

        q_values = self.model(states)
        # print("Q-values shape:", q_values.shape)
        
        next_q_values = self.model(next_states)
        next_q_actions = torch.argmax(next_q_values, dim=1)
        target_next_q_values = self.target_model(next_states)
        next_q_values = target_next_q_values.gather(1, next_q_actions.unsqueeze(1)).squeeze(1)

        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        targets = rewards + (self.gamma * next_q_values * (1 - done))

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.update_target_frequency == 0:
            self.update_target_model()

        # print(f"Replay Step: {self.update_counter}, Loss: {loss.item()}")

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def set_eval_mode(self):
        self.model.eval()


class MultiAgentGraph:
    def __init__(self, machines, jobs):
        self.machines = machines
        self.jobs = jobs
        self.graph = self.build_graph()

    def build_graph(self):
        graph = {}
        # Initialize machine nodes
        for machine in self.machines:
            graph[f'M_{machine}'] = {
                'type': 'machine',
                'static_edges': [],
                'dynamic_edges': {'executing': [], 'waiting': [], 'routing': []}
            }
        # Initialize job nodes
        for job in self.jobs:
            graph[f'J_{job}'] = {
                'type': 'job',
                'dynamic_edges': {'executing': [], 'waiting': [], 'routing': []}
            }
        return graph

    def update_graph(self, state, op=None, machine=None):
        # print("Updating graph")
        # Clear dynamic edges
        for node in self.graph.values():
            if node['type'] == 'machine':
                node['dynamic_edges'] = {'executing': [], 'waiting': [], 'routing': []}
            elif node['type'] == 'job':
                node['dynamic_edges'] = {'executing': [], 'waiting': [], 'routing': []}

        # Update dynamic edges based on current state
        for job_id, job_info in state['jobs'].items():
            job_key = job_id  # 이미 J_ 형태로 주어진 key
            # print(f"Processing job: {job_key}")
            current_machine = job_info['current_machine']
            if current_machine is not None:
                machine_key = f'M_{current_machine}'
                # print(f"Job {job_key} is currently executing on machine {machine_key}")
                self.graph[machine_key]['dynamic_edges']['executing'].append(job_key)
                self.graph[job_key]['dynamic_edges']['executing'].append(machine_key)
            for machine_id in job_info['waiting_machines']:
                machine_key = f'M_{machine_id}'
                # print(f"Job {job_key} is waiting on machine {machine_key}")
                self.graph[machine_key]['dynamic_edges']['waiting'].append(job_key)
                self.graph[job_key]['dynamic_edges']['waiting'].append(machine_key)
            for machine_id in job_info['routing_machines']:
                machine_key = f'M_{machine_id}'
                # print(f"Job {job_key} is routing on machine {machine_key}")
                self.graph[machine_key]['dynamic_edges']['routing'].append(job_key)
                self.graph[job_key]['dynamic_edges']['routing'].append(machine_key)

        # Print the updated graph for debugging
        # print("Updated graph:")
        # for node, data in self.graph.items():
        #     print(f"{node}: {data}")


    def get_state_features(self, agent_id):
        if self.graph[agent_id]['type'] == 'machine':
            features = self.get_machine_features(agent_id)
        elif self.graph[agent_id]['type'] == 'job':
            features = self.get_job_features(agent_id)
        
        # Convert the features dictionary to a list
        features_list = list(features.values())
        return features_list

    def get_machine_features(self, machine_id):
        machine_node = self.graph[machine_id]
        features = {
            'executing_jobs': len(machine_node['dynamic_edges']['executing']),
            'waiting_jobs': len(machine_node['dynamic_edges']['waiting']),
            'routing_jobs': len(machine_node['dynamic_edges']['routing'])
        }
        return features

    def get_job_features(self, job_id):
        job_node = self.graph[job_id]
        features = {
            'executing_machines': len(job_node['dynamic_edges']['executing']),
            'waiting_machines': len(job_node['dynamic_edges']['waiting']),
            'routing_machines': len(job_node['dynamic_edges']['routing'])
        }
        return features

class SupervisorAgent(DDQNAgent):
    def __init__(self, input_dim, action_size, update_target_frequency=80, replay_start_size=100, batch_size=80):
        super().__init__(input_dim, action_size, update_target_frequency, replay_start_size, batch_size)
        self.used_machines = set()

    def update(self, machine_id):
        self.used_machines.add(machine_id)

    def is_machine_used(self, machine_id):
        return machine_id in self.used_machines

    def reset(self):
        self.used_machines.clear()

    def remember(self, state, action, reward, next_state, done):
        # action이 시퀀스인 경우 첫 번째 요소만 저장
        if isinstance(action, (list, tuple)):
            action = action[0]
        self.memory.append((state, action, reward, next_state, done))

    def select_machine(self, available_machines, state_features):
        if np.random.rand() <= self.epsilon:
            return random.choice(available_machines)
        
        machine_values = []
        for machine in available_machines:
            machine_state = torch.FloatTensor(state_features[f'M_{machine}']).unsqueeze(0)
            
            # 현재 네트워크로 행동(machine) 선택
            q_values = self.model(machine_state)
            best_action = q_values.argmax(dim=1)
            
            # 타겟 네트워크로 선택된 행동의 가치 평가
            target_q_values = self.target_model(machine_state)
            machine_value = target_q_values[0, best_action].item()
            
            machine_values.append(machine_value)
        
        return available_machines[np.argmax(machine_values)]

    def replay(self):
        if len(self.memory) < self.replay_start_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, done = zip(*minibatch)

        states = torch.FloatTensor(np.array(states))
        
        # actions 디버깅
        # print("Actions before conversion:", actions)
        actions = [action if isinstance(action, (int, np.integer)) else action[0] for action in actions]
        # print("Actions after conversion:", actions)
        
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        done = torch.FloatTensor(done)

        q_values = self.model(states)
        next_q_values = self.model(next_states)
        next_q_actions = torch.argmax(next_q_values, dim=1)
        target_next_q_values = self.target_model(next_states)
        next_q_values = target_next_q_values.gather(1, next_q_actions.unsqueeze(1)).squeeze(1)

        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        targets = rewards + (self.gamma * next_q_values * (1 - done))

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.update_target_frequency == 0:
            self.update_target_model()

        # print(f"Replay Step: {self.update_counter}, Loss: {loss.item()}")


class MultiAgentSystem:
    def __init__(self, machines, jobs, state_size, action_size):
        self.graph = MultiAgentGraph(machines, jobs)
        self.machines = machines
        self.jobs = jobs
        self.action_size = action_size
        self.agents = {}
        self.replay_start_size = 100
        self.global_episode_count = 0
        self.supervisor = None  # SupervisorAgent 초기화 위치 변경
        self.actual_actions = {}  # 실제 액션을 저장하기 위한 딕셔너리

    def initialize_agents(self, sample_state):
        input_dim = self.calculate_input_dim(sample_state)
        self.agents = {f'M_{machine}': DDQNAgent(input_dim, self.action_size) for machine in self.machines}
        self.agents.update({f'J_{job}': DDQNAgent(input_dim, self.action_size) for job in self.jobs})
        self.supervisor = SupervisorAgent(input_dim, self.action_size)  # SupervisorAgent 초기화
        self.replay_start_size = 100
        self.global_episode_count = 0

    def reset(self):
        for agent in self.agents.values():
            agent.epsilon = 0.99  # 에이전트의 탐험률 초기화 (필요에 따라 조정)
        self.supervisor.reset()  # SupervisorAgent 초기화

    def calculate_input_dim(self, state):
        graph_features_length = len(self.graph.get_state_features(f'M_{self.machines[0]}'))
        env_state_length = (
            1 +
            len(state['job_completion']) +
            len(state['machine_available_time']) +
            len(state['machine_utilization']) +
            len(state['job_queue_length']) +
            len(state['job_progress']) +
            len(state['remaining_job_time']) +
            sum(len(status) for status in state['job_op_status']) +
            len(state['machine_status'])
        )
        return graph_features_length + env_state_length

    def act(self, state, mask):
        actions = {}
        for agent_id in self.agents:
            features = self.graph.get_state_features(agent_id)
            actions[agent_id] = self.agents[agent_id].act(features, mask)
        return actions

    def remember(self, agent_id, state, action, reward, next_state, done):
        if isinstance(action, dict):
            action = next(iter(action.values()))
        self.agents[agent_id].remember(state, action, reward, next_state, done)

    def replay(self):
        for agent_id, agent in self.agents.items():
            if len(agent.memory) >= agent.replay_start_size:
                agent.replay()
        self.supervisor.replay()  # Supervisor의 replay도 호출

    def update_target_models(self):
        for agent in self.agents.values():
            agent.update_target_model()

    def load(self, names):
        for agent_id, name in names.items():
            self.agents[agent_id].load(name)

    def save(self, names):
        for agent_id, name in names.items():
            self.agents[agent_id].save(name)

    def memory_size(self):
        return sum(len(agent.memory) for agent in self.agents.values())

    def end_episode(self, state_features, reward_task, reward_machine, done):
        self.global_episode_count += 1
        for agent_id, agent in self.agents.items():
            actual_action = self.actual_actions.get(agent_id, 0)
            # actual_action이 딕셔너리인 경우 값만 전달
            if isinstance(actual_action, dict):
                actual_action = next(iter(actual_action.values()))
            agent.remember(state_features[agent_id], actual_action, reward_task if agent_id.startswith('J_') else reward_machine, state_features[agent_id], done)
            agent.update_epsilon(self.global_episode_count)
        
        # Supervisor의 실제 액션 저장
        supervisor_action = self.actual_actions.get('supervisor', 0)
        if isinstance(supervisor_action, dict):
            supervisor_action = next(iter(supervisor_action.values()))
        self.supervisor.remember(state_features['supervisor'], supervisor_action, reward_machine, state_features['supervisor'], done)
        
        if self.memory_size() >= self.replay_start_size:
            self.replay()

def train_individual_models(datasets, num_episodes_per_dataset):
    for i, dataset in enumerate(datasets):
        print(f"Starting training for dataset {dataset} ({i+1}/{len(datasets)})")

        data = RLDataset_FJSSP(dataset)
        
        action_size = data.total_operations
        process_times = [[(m, t) for m, t in ops] for job in data.op_data for ops in job]
        machine_sequence = [[[(m, t) for m, t in ops] for ops in job] for job in data.op_data]

        env = JobShopEnv_FJSSP(process_times, machine_sequence, solutions=None)
        initial_state = env.reset()

        # `initialize_agents` 메서드를 통해 `MultiAgentSystem`과 `SupervisorAgent` 초기화
        multi_agent_system = MultiAgentSystem(machines=range(data.n_machine), jobs=range(data.n_job), state_size=None, action_size=action_size)
        multi_agent_system.initialize_agents(initial_state)

        for episode in range(num_episodes_per_dataset):
            state = env.reset()
            multi_agent_system.reset()
            done = False
            episode_reward = 0
            step_count = 0

            while not done:
                state = env.get_state()
                multi_agent_system.graph.update_graph(state)
                state_features = {}
                for agent_id in multi_agent_system.agents:
                    graph_features = multi_agent_system.graph.get_state_features(agent_id)
                    combined_features = np.concatenate((
                        graph_features,
                        np.array([state['current_time']]),
                        state['job_completion'],
                        state['machine_available_time'],
                        state['machine_utilization'],
                        state['job_queue_length'],
                        state['job_progress'],
                        state['remaining_job_time'],
                        np.array([status for job_status in state['job_op_status'] for status in job_status]),
                        np.array(state['machine_status'], dtype=int)
                    ))
                    state_features[agent_id] = combined_features

                # 랜덤으로 machine agent 선택
                random_machine_agent = random.choice(list(multi_agent_system.agents.keys()))
                supervisor_graph_features = multi_agent_system.graph.get_state_features(random_machine_agent)

                supervisor_state_features = np.concatenate((
                    supervisor_graph_features,
                    np.array([state['current_time']]),
                    state['job_completion'],
                    state['machine_available_time'],
                    state['machine_utilization'],
                    state['job_queue_length'],
                    state['job_progress'],
                    state['remaining_job_time'],
                    np.array([status for job_status in state['job_op_status'] for status in job_status]),
                    np.array(state['machine_status'], dtype=int)
                ))
                state_features['supervisor'] = supervisor_state_features

                valid_actions = env.get_valid_actions()
                machine_actions = {}
                available_machines = [m for job, op, m in valid_actions if not multi_agent_system.supervisor.is_machine_used(m)]
                
                if not available_machines:
                    multi_agent_system.supervisor.reset()
                    available_machines = [m for job, op, m in valid_actions]

                # SupervisorAgent가 machine 선택
                selected_machine = multi_agent_system.supervisor.select_machine(available_machines, state_features)
                
                # 선택된 machine에 대해서만 action 결정
                selected_machine_agent_id = f'M_{selected_machine}'
                valid_operations = []
                for job, op, machine in valid_actions:
                    if machine == selected_machine:
                        valid_operations.append((job, op))

                if valid_operations:
                    selected_operation = multi_agent_system.agents[selected_machine_agent_id].act(
                        state_features[selected_machine_agent_id], 
                        valid_operations
                    )
                    selected_job, selected_op = selected_operation
                    action = (selected_job, selected_op, selected_machine)
                    machine_actions[action] = multi_agent_system.agents[selected_machine_agent_id].model(
                        torch.FloatTensor(state_features[selected_machine_agent_id]).unsqueeze(0)
                    ).squeeze(0)[valid_operations.index(selected_operation)].item()

                if not machine_actions:
                    continue

                best_action = max(machine_actions, key=machine_actions.get)
                selected_job, selected_op, selected_machine = best_action
                multi_agent_system.supervisor.update(selected_machine)

                next_state, done, step_reward = env.step(selected_job, selected_op, selected_machine)
                multi_agent_system.graph.update_graph(next_state)
                next_state_features = {}
                for agent_id in multi_agent_system.agents:
                    graph_features = multi_agent_system.graph.get_state_features(agent_id)
                    combined_features = np.concatenate((
                        graph_features,
                        np.array([next_state['current_time']]),
                        next_state['job_completion'],
                        next_state['machine_available_time'],
                        next_state['machine_utilization'],
                        next_state['job_queue_length'],
                        next_state['job_progress'],
                        next_state['remaining_job_time'],
                        np.array([status for job_status in next_state['job_op_status'] for status in job_status]),
                        np.array(next_state['machine_status'], dtype=int)
                    ))
                    next_state_features[agent_id] = combined_features

                # SupervisorAgent의 상태 추가
                next_supervisor_state_features = np.concatenate((
                    supervisor_graph_features,
                    np.array([next_state['current_time']]),
                    next_state['job_completion'],
                    next_state['machine_available_time'],
                    next_state['machine_utilization'],
                    next_state['job_queue_length'],
                    next_state['job_progress'],
                    next_state['remaining_job_time'],
                    np.array([status for job_status in next_state['job_op_status'] for status in job_status]),
                    np.array(next_state['machine_status'], dtype=int)
                ))
                next_state_features['supervisor'] = next_supervisor_state_features

                multi_agent_system.remember(selected_machine_agent_id, state_features[selected_machine_agent_id], machine_actions[action], step_reward, next_state_features[selected_machine_agent_id], done)
                multi_agent_system.supervisor.remember(supervisor_state_features, best_action, step_reward, next_supervisor_state_features, done)  # Supervisor도 기억

                state = next_state
                episode_reward += step_reward
                step_count += 1
                if step_count % 50 == 0:
                    gc.collect()
                    print(f"Step {step_count} completed. Memory cleaned.")

                if done:
                    reward_task, reward_machine = env.calculate_episode_rewards()
                    reward_task += episode_reward
                    reward_machine += episode_reward
                    print(f"Training: Episode {episode+1} processed. Reward Task: {reward_task}, Reward Machine: {reward_machine}, episode reward every step: {episode_reward}")
                    break

            multi_agent_system.end_episode(state_features, reward_task, reward_machine, done)
            
            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes_per_dataset} completed for dataset {dataset}")
                gc.collect()

        print(f"Completed training for dataset {dataset} ({i+1}/{len(datasets)})")
        gc.collect()

    agent_save_paths = {agent_id: f"{agent_id}.pth" for agent_id in multi_agent_system.agents.keys()}
    multi_agent_system.save(agent_save_paths)

    return multi_agent_system, None, data.n_machine


def predict(multi_agent_system, env, test_dataset, num_predictions=1, max_steps=100000000, epsilon=0.001):
    valid_solution_count = 0
    all_predictions = []
    step_count = 0

    for agent in multi_agent_system.agents.values():
        agent.epsilon = epsilon
        agent.set_eval_mode()
    multi_agent_system.supervisor.epsilon = epsilon
    multi_agent_system.supervisor.set_eval_mode()

    while valid_solution_count < num_predictions and step_count < max_steps:
        state = env.reset()
        solution = []
        done = False
        episode_reward = 0
        multi_agent_system.reset()

        while not done and step_count < max_steps:
            state = env.get_state()
            multi_agent_system.graph.update_graph(state)
            state_features = {}
            for agent_id in multi_agent_system.agents:
                graph_features = multi_agent_system.graph.get_state_features(agent_id)
                combined_features = np.concatenate((
                    graph_features,
                    np.array([state['current_time']]),
                    state['job_completion'],
                    state['machine_available_time'],
                    state['machine_utilization'],
                    state['job_queue_length'],
                    state['job_progress'],
                    state['remaining_job_time'],
                    np.array([status for job_status in state['job_op_status'] for status in job_status]),
                    np.array(state['machine_status'], dtype=int)
                ))
                state_features[agent_id] = combined_features

            # 랜덤으로 machine agent 선택
            random_machine_agent = random.choice(list(multi_agent_system.agents.keys()))
            supervisor_graph_features = multi_agent_system.graph.get_state_features(random_machine_agent)
            
            supervisor_state_features = np.concatenate((
                supervisor_graph_features,
                np.array([state['current_time']]),
                state['job_completion'],
                state['machine_available_time'],
                state['machine_utilization'],
                state['job_queue_length'],
                state['job_progress'],
                state['remaining_job_time'],
                np.array([status for job_status in state['job_op_status'] for status in job_status]),
                np.array(state['machine_status'], dtype=int)
            ))
            state_features['supervisor'] = supervisor_state_features

            valid_actions = env.get_valid_actions()
            machine_actions = {}
            available_machines = [m for job, op, m in valid_actions if not multi_agent_system.supervisor.is_machine_used(m)]
            
            if not available_machines:
                multi_agent_system.supervisor.reset()
                available_machines = [m for job, op, m in valid_actions]

            # SupervisorAgent가 machine 선택
            selected_machine = multi_agent_system.supervisor.select_machine(available_machines, state_features)
            
            # 선택된 machine에 대해서만 action 결정
            selected_machine_agent_id = f'M_{selected_machine}'
            valid_operations = []
            for job, op, machine in valid_actions:
                if machine == selected_machine:
                    valid_operations.append((job, op))

            if valid_operations:
                selected_operation = multi_agent_system.agents[selected_machine_agent_id].act(
                    state_features[selected_machine_agent_id], 
                    valid_operations
                )
                selected_job, selected_op = selected_operation
                action = (selected_job, selected_op, selected_machine)
                machine_actions[action] = multi_agent_system.agents[selected_machine_agent_id].model(
                    torch.FloatTensor(state_features[selected_machine_agent_id]).unsqueeze(0)
                ).squeeze(0)[valid_operations.index(selected_operation)].item()

            if not machine_actions:
                continue

            best_action = max(machine_actions, key=machine_actions.get)
            selected_job, selected_op, selected_machine = best_action
            multi_agent_system.supervisor.update(selected_machine)

            next_state, done, step_reward = env.step(selected_job, selected_op, selected_machine)
            multi_agent_system.graph.update_graph(next_state)
            next_state_features = {}
            for agent_id in multi_agent_system.agents:
                graph_features = multi_agent_system.graph.get_state_features(agent_id)
                combined_features = np.concatenate((
                    graph_features,
                    np.array([next_state['current_time']]),
                    next_state['job_completion'],
                    next_state['machine_available_time'],
                    next_state['machine_utilization'],
                    next_state['job_queue_length'],
                    next_state['job_progress'],
                    next_state['remaining_job_time'],
                    np.array([status for job_status in next_state['job_op_status'] for status in job_status]),
                    np.array(next_state['machine_status'], dtype=int)
                ))
                next_state_features[agent_id] = combined_features

            # SupervisorAgent의 상태 추가
            next_supervisor_state_features = np.concatenate((
                supervisor_graph_features,
                np.array([next_state['current_time']]),
                next_state['job_completion'],
                next_state['machine_available_time'],
                next_state['machine_utilization'],
                next_state['job_queue_length'],
                next_state['job_progress'],
                next_state['remaining_job_time'],
                np.array([status for job_status in next_state['job_op_status'] for status in job_status]),
                np.array(next_state['machine_status'], dtype=int)
            ))
            next_state_features['supervisor'] = next_supervisor_state_features

            multi_agent_system.remember(selected_machine_agent_id, state_features[selected_machine_agent_id], machine_actions[action], step_reward, next_state_features[selected_machine_agent_id], done)
            multi_agent_system.supervisor.remember(supervisor_state_features, best_action, step_reward, next_supervisor_state_features, done)

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
            reward_task += episode_reward
            reward_machine += episode_reward
            multi_agent_system.end_episode(state_features, reward_task, reward_machine, done)
            print(f"Prediction: Episode {valid_solution_count+1}/{num_predictions} processed. "
                  f"Reward Task: {reward_task}, Reward Machine: {reward_machine}, Episode every step Reward: {episode_reward}")
            valid_solution_count += 1
            all_predictions.append(solution)
            gc.collect()

    if valid_solution_count != num_predictions:
        print(f"Warning: Only {valid_solution_count} valid solutions were collected.")

    return all_predictions
def main():
    datasets = [
        'fjsspdataset/HurinkRdata7.fjs',
    ]
    num_episodes_per_dataset = 300

    multi_agent_system, max_state_size, action_size = train_individual_models(datasets, num_episodes_per_dataset)

    test_dataset = RLDataset_FJSSP('fjsspdataset/HurinkRdata7.fjs')

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
