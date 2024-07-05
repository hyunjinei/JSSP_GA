import os
import pandas as pd
from RLDataset_FJSSP import RLDataset_FJSSP
from JobShopEnv_FJSSP_machine import JobShopEnv_FJSSP
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


def pad_state_action(state, action, max_state_size=1400, pad_value=-2):
    if len(state) > max_state_size:
        state = state[:max_state_size]

    max_action_size = 20 * 10
    padded_state = np.full(max_state_size, pad_value)
    padded_state[:len(state)] = state
    mask = np.zeros(max_state_size)
    mask[:len(state)] = 1
    padded_action = np.full(max_action_size, pad_value)
    padded_action[:len(action)] = action
    return padded_state, torch.tensor(mask, dtype=torch.float32)

class DQN(nn.Module):
    def __init__(self, max_state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(max_state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, action_size)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(0)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x


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

    def update_graph(self, state, op, machine):
        # Clear dynamic edges
        for node in self.graph.values():
            if node['type'] == 'machine':
                node['dynamic_edges'] = {'executing': [], 'waiting': [], 'routing': []}
            elif node['type'] == 'job':
                node['dynamic_edges'] = {'executing': [], 'waiting': [], 'routing': []}

        # Update dynamic edges based on current state
        for job_id, job_info in state['jobs'].items():
            current_machine = job_info['current_machine']
            if current_machine is not None:
                self.graph[current_machine]['dynamic_edges']['executing'].append(job_id)
                self.graph[job_id]['dynamic_edges']['executing'].append(current_machine)
            for machine_id in job_info['waiting_machines']:
                self.graph[machine_id]['dynamic_edges']['waiting'].append(job_id)
                self.graph[job_id]['dynamic_edges']['waiting'].append(machine_id)
            for machine_id in job_info['routing_machines']:
                self.graph[machine_id]['dynamic_edges']['routing'].append(job_id)
                self.graph[job_id]['dynamic_edges']['routing'].append(machine_id)

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

class MultiAgentSystem:
    def __init__(self, machines, jobs, state_size, action_size):
        self.graph = MultiAgentGraph(machines, jobs)
        self.agents = {f'M_{machine}': DQNAgent(state_size, action_size) for machine in machines}
        self.agents.update({f'J_{job}': DQNAgent(state_size, action_size) for job in jobs})
        self.replay_start_size = next(iter(self.agents.values())).replay_start_size

    def act(self, state, mask):
        actions = {}
        for agent_id in self.agents:
            features = self.graph.get_state_features(agent_id)
            features, _ = pad_state_action(features, [])
            actions[agent_id] = self.agents[agent_id].act(features, mask)
        return actions

    def remember(self, agent_id, state, action, reward, next_state, done):
        self.agents[agent_id].remember(state, action[agent_id], reward, next_state, done)

    def replay(self):
        for agent in self.agents.values():
            agent.replay()

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

    def update_epsilon(self):
        for agent in self.agents.values():
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

class DQNAgent:
    def __init__(self, state_size, action_size, update_target_frequency=32, replay_start_size=1000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000000)
        self.gamma = 0.99
        self.epsilon = 0.99
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()
        self.update_target_frequency = update_target_frequency
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size
        self.update_counter = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, mask=None):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(torch.tensor(state, dtype=torch.float32).unsqueeze(0), mask)
        return torch.argmax(act_values, dim=1).item()

    def replay(self):
        if len(self.memory) < self.replay_start_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, done = zip(*minibatch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        done = torch.FloatTensor(done)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = next_q_values.max(1)[0]
        targets = rewards + (self.gamma * next_q_values * (1 - done))

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.update_counter += 1
        if self.update_counter % self.update_target_frequency == 0:
            self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def set_eval_mode(self):
        self.model.eval()


def train_individual_models(datasets, num_episodes_per_dataset):
    max_state_size = 20 * 10 * 7
    action_size = 20 * 10

    multi_agent_system = MultiAgentSystem(machines=range(5), jobs=range(10), state_size=max_state_size, action_size=action_size)

    for i, dataset in enumerate(datasets):
        print(f"Starting training for dataset {dataset} ({i+1}/{len(datasets)})")

        data = RLDataset_FJSSP(dataset)

        process_times = [[(m, t) for m, t in ops] for job in data.op_data for ops in job]
        machine_sequence = [[[(m, t) for m, t in ops] for ops in job] for job in data.op_data]

        env = JobShopEnv_FJSSP(process_times, machine_sequence, solutions=None)

        for episode in range(num_episodes_per_dataset):
            state = env.reset()
            done = False
            episode_reward = 0
            step_count = 0

            while not done:
                task_state = np.concatenate((
                    np.array([state['current_time']]),
                    state['job_completion'],
                    state['machine_available_time'],
                    state['job_progress'],
                    state['remaining_job_time'],
                    np.array([status for job_status in state['job_op_status'] for status in job_status]),
                    np.array(state['machine_status'], dtype=int)
                ))
                task_state, task_mask = pad_state_action(task_state, [])
                task_actions = multi_agent_system.act(task_state, task_mask)

                selected_job, selected_op = divmod(task_actions[f'J_{np.random.randint(0, 10)}'], env.n_machines)

                machine_state = np.concatenate((
                    np.array([state['current_time']]),
                    state['job_completion'],
                    state['machine_available_time'],
                    state['machine_utilization'],
                    state['job_queue_length'],
                    np.array([status for job_status in state['job_op_status'] for status in job_status]),
                    np.array(state['machine_status'], dtype=int)
                ))
                machine_state, machine_mask = pad_state_action(machine_state, [])
                machine_actions = multi_agent_system.act(machine_state, machine_mask)

                selected_machine = machine_actions[f'M_{np.random.randint(0, 5)}'] % env.n_machines

                valid_actions = env.get_valid_actions()
                if (selected_job, selected_op, selected_machine) not in valid_actions:
                    continue

                next_state, done, step_reward = env.step(selected_job, selected_op, selected_machine)

                # Print the current state, next state, and other relevant details
                # print("Current State:", state)
                # print("Next State:", next_state)
                # print("Selected Job:", selected_job)
                # print("Selected Operation:", selected_op)
                # print("Selected Machine:", selected_machine)
                # print("Step Reward:", step_reward)
                # print("Done:", done)

                next_task_state = np.concatenate((
                    np.array([next_state['current_time']]),
                    next_state['job_completion'],
                    next_state['machine_available_time'],
                    next_state['job_progress'],
                    next_state['remaining_job_time'],
                    np.array([status for job_status in next_state['job_op_status'] for status in job_status]),
                    np.array(next_state['machine_status'], dtype=int)
                ))
                next_task_state, _ = pad_state_action(next_task_state, [])
                multi_agent_system.remember(f'J_{selected_job}', state, task_actions, step_reward, next_state, done)

                next_machine_state = np.concatenate((
                    np.array([next_state['current_time']]),
                    next_state['job_completion'],
                    next_state['machine_available_time'],
                    next_state['machine_utilization'],
                    next_state['job_queue_length'],
                    np.array([status for job_status in next_state['job_op_status'] for status in job_status]),
                    np.array(next_state['machine_status'], dtype=int)
                ))
                next_machine_state, _ = pad_state_action(next_machine_state, [])
                multi_agent_system.remember(f'M_{selected_machine}', state, machine_actions, step_reward, next_state, done)

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

            if multi_agent_system.memory_size() >= multi_agent_system.replay_start_size:
                for _ in range(2):
                    multi_agent_system.replay()

            multi_agent_system.update_epsilon()

            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes_per_dataset} completed for dataset {dataset}")
                gc.collect()

        print(f"Completed training for dataset {dataset} ({i+1}/{len(datasets)})")
        gc.collect()

    agent_save_paths = {agent_id: f"{agent_id}.pth" for agent_id in multi_agent_system.agents.keys()}
    multi_agent_system.save(agent_save_paths)

    return multi_agent_system, max_state_size, action_size

def predict(multi_agent_system, env, test_dataset, num_predictions=1, max_steps=100000000):
    valid_solution_count = 0
    all_predictions = []
    step_count = 0
    for agent_id in multi_agent_system.agents:
        multi_agent_system.agents[agent_id].load(f'{agent_id}.pth')
    for agent in multi_agent_system.agents.values():
        agent.set_eval_mode()

    while valid_solution_count < num_predictions and step_count < max_steps:
        state = env.reset()
        solution = []
        done = False
        episode_reward = 0

        while not done and step_count < max_steps:
            task_state = np.concatenate((
                np.array([state['current_time']]),
                state['job_completion'],
                state['machine_available_time'],
                state['job_progress'],
                state['remaining_job_time'],
                np.array([status for job_status in state['job_op_status'] for status in job_status]),
                np.array(state['machine_status'], dtype=int)
            ))
            task_state, task_mask = pad_state_action(task_state, [])
            task_actions = multi_agent_system.act(task_state, task_mask)

            selected_job, selected_op = divmod(task_actions[f'J_{np.random.randint(0, 10)}'], env.n_machines)

            machine_state = np.concatenate((
                np.array([state['current_time']]),
                state['job_completion'],
                state['machine_available_time'],
                state['machine_utilization'],
                state['job_queue_length'],
                np.array([status for job_status in state['job_op_status'] for status in job_status]),
                np.array(state['machine_status'], dtype=int)
            ))
            machine_state, machine_mask = pad_state_action(machine_state, [])
            machine_actions = multi_agent_system.act(machine_state, machine_mask)

            selected_machine = machine_actions[f'M_{np.random.randint(0, 5)}'] % env.n_machines

            if selected_job is None or selected_op is None or selected_machine is None:
                continue

            valid_actions = env.get_valid_actions()
            if (selected_job, selected_op, selected_machine) not in valid_actions:
                continue

            next_state, done, step_reward = env.step(selected_job, selected_op, selected_machine)
            duration = next(t for m, t in env.machine_sequence[selected_job][selected_op] if m == selected_machine)
            solution.append((selected_job, selected_op, selected_machine, duration))
            step_count += 1
            episode_reward += step_reward

            if step_count % 100 == 0:
                print(f"Predicting: Step {step_count}, Current Solution: {solution[-5:]}")
                gc.collect()

            state = next_state
            print(f"Step {step_count}")
            print(f"Step reward: {step_reward}")

        if done:
            reward_task, reward_machine = env.calculate_episode_rewards()
            reward_task += episode_reward
            reward_machine += episode_reward
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
    num_episodes_per_dataset = 2

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
    all_predictions = predict(multi_agent_system, env, test_dataset, num_predictions=1, max_steps=100000000)

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
