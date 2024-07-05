import os
import pandas as pd
from RLDataset_FJSSP import RLDataset_FJSSP
from JobShopEnv_FJSSP import JobShopEnv_FJSSP
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

class DQN(nn.Module):
    def __init__(self, input_size, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Swap dimensions for LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last output of LSTM
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, update_target_frequency=32, replay_start_size=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000000)
        self.gamma = 0.1 # 할인율 (미래 보상의 현재 가치)
        self.epsilon = 0.9  # 탐험(exploration) 확률 초기값
        self.epsilon_min = 0.01  # 탐험 확률의 최소값
        self.epsilon_decay = 0.995  # 탐험 확률 감소율
        self.learning_rate = 0.0001  # 학습률 0.0001
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = F.smooth_l1_loss
        self.update_target_frequency = update_target_frequency
        self.replay_start_size = replay_start_size
        self.update_counter = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        print("Updated target network.")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, mask=None, env=None):
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state).detach().numpy()[0]
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(act_values)

    def replay(self, batch_size):
        if len(self.memory) < self.replay_start_size:
            return  

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(current_q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        logging.info(f"Replay: Batch processed. Loss: {loss.item()}")
        print(f"Replay: Batch processed. Loss: {loss.item()}")

        self.update_counter += 1

        if self.update_counter % self.update_target_frequency == 0:
            self.update_target_model()
            print("Updated target network after replay.") 

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

class TaskAgent(DQNAgent):
    def __init__(self, state_size, action_size, update_target_frequency=32, replay_start_size=3000):
        super(TaskAgent, self).__init__(state_size, action_size, update_target_frequency, replay_start_size)

    def act(self, state, mask=None, env=None):
        return super(TaskAgent, self).act(state, mask, env)

class MachineAgent(DQNAgent):
    def __init__(self, state_size, action_size, update_target_frequency=32, replay_start_size=3000):
        super(MachineAgent, self).__init__(state_size, action_size, update_target_frequency, replay_start_size)

    def act(self, state, mask=None, env=None):
        return super(MachineAgent, self).act(state, mask, env)

def train_individual_models(datasets, num_episodes_per_dataset):
    for i, dataset in enumerate(datasets):
        print(f"Starting training for dataset {dataset} ({i+1}/{len(datasets)})")

        data = RLDataset_FJSSP(dataset)

        process_times = [[(m, t) for m, t in ops] for job in data.op_data for ops in job]
        machine_sequence = [[[(m, t) for m, t in ops] for ops in job] for job in data.op_data]

        env = JobShopEnv_FJSSP(process_times, machine_sequence, solutions=None)

        # 첫 번째 상태를 가져와서 크기를 결정
        initial_state = env.reset()
        state_size = len(np.concatenate(([initial_state[0]], initial_state[1], initial_state[2], initial_state[3], initial_state[4])))
        action_size = env.n_jobs * env.n_machines

        # 데이터셋마다 새로운 에이전트 생성
        task_agent = TaskAgent(state_size, action_size, update_target_frequency=128, replay_start_size=100)
        machine_agent = MachineAgent(state_size, action_size, update_target_frequency=128, replay_start_size=100)

        for job in range(len(machine_sequence)):
            print(f"Job {job}: {machine_sequence[job]}")

        for episode in range(num_episodes_per_dataset):
            state = env.reset()
            state = np.concatenate(([state[0]], state[1], state[2], state[3], state[4]))
            max_op_counts = [len(ops) for ops in machine_sequence]
            job_op_counter = [0] * env.n_jobs

            done = False
            episode_reward = 0
            while not done:
                task_action = task_agent.act(state, None, env)
                selected_job, selected_op = divmod(task_action, env.n_machines)

                if selected_job >= env.n_jobs or selected_op >= max_op_counts[selected_job] or job_op_counter[selected_job] != selected_op:
                    continue

                machine_action = machine_agent.act(state, None, env)
                selected_machine = machine_action % env.n_machines

                machine_options = env.machine_sequence[selected_job][selected_op]
                valid_machines = [m for m, t in machine_options]
                if selected_machine not in valid_machines:
                    continue

                next_state, done, step_reward = env.step(selected_job, selected_op, selected_machine)
                next_state = np.concatenate(([next_state[0]], next_state[1], next_state[2], next_state[3], next_state[4]))

                task_agent.remember(state, selected_job * env.n_machines + selected_op, step_reward, next_state, done)
                machine_agent.remember(state, selected_op * env.n_jobs + selected_job, step_reward, next_state, done)

                state = next_state

                job_op_counter[selected_job] += 1
                if job_op_counter[selected_job] > max_op_counts[selected_job]:
                    job_op_counter[selected_job] = max_op_counts[selected_job]

                episode_reward += step_reward

                if done or all(count >= max_op_counts[j] for j, count in enumerate(job_op_counter)):
                    reward_task, reward_machine = env.calculate_episode_rewards()
                    reward_task += episode_reward
                    reward_machine += episode_reward
                    print(f"Training: Episode {episode+1} processed. Reward Task: {reward_task}, Reward Machine: {reward_machine}, episode reward every step: {episode_reward}")
                    break

            if len(task_agent.memory) >= task_agent.replay_start_size:
                for _ in range(2):
                    task_agent.replay(64)
                    machine_agent.replay(64)

            if task_agent.update_counter % task_agent.update_target_frequency == 0:
                task_agent.update_target_model()
                print("Updated target network for task agent during training.")
            
            if machine_agent.update_counter % machine_agent.update_target_frequency == 0:
                machine_agent.update_target_model()
                print("Updated target network for machine agent during training.")

            if episode % 1 == 0:
                print(f"Episode {episode}/{num_episodes_per_dataset} completed for dataset {dataset}")

        print(f"Completed training for dataset {dataset} ({i+1}/{len(datasets)})")

    return task_agent, machine_agent, state_size, action_size

def predict(task_agent, machine_agent, env, test_dataset, num_predictions=1, max_steps=100000000):
    valid_solution_count = 0
    all_predictions = []
    step_count = 0

    while valid_solution_count < num_predictions and step_count < max_steps:
        state = env.reset()
        max_op_counts = test_dataset.max_op_counts.copy()
        job_op_counter = {job: 0 for job in range(env.n_jobs)}

        state = np.concatenate(([state[0]], state[1], state[2], state[3], state[4]))

        solution = []
        done = False
        episode_reward = 0

        print(f"Initial max_op_counts: {max_op_counts}")

        while not done and step_count < max_steps:
            task_action = task_agent.act(state, None, env)
            job, op = divmod(task_action, env.n_machines)

            if job >= env.n_jobs or op >= len(env.machine_sequence[job]) or job_op_counter[job] != op:
                continue

            machine_action = machine_agent.act(state, None, env)
            machine = machine_action % env.n_machines

            machine_options = env.machine_sequence[job][op]
            machine_duration = next((t for m, t in machine_options if m == machine), None)
            if machine_duration is None:
                continue

            next_state, done, step_reward = env.step(job, op, machine)
            solution.append((job, op, machine, machine_duration))
            job_op_counter[job] += 1
            step_count += 1
            episode_reward += step_reward

            if job_op_counter[job] > max_op_counts[job]:
                max_op_counts[job] = -1
                print(f"Job {job} completed. Updated max_op_counts: {max_op_counts}")

            if step_count % 100 == 0:
                print(f"Predicting: Step {step_count}, Current Solution: {solution[-5:]}")
                gc.collect()

            next_state = np.concatenate(([next_state[0]], next_state[1], next_state[2], next_state[3], next_state[4]))
            
            state = next_state
            print(f"Step {step_count}, job_op_counter: {job_op_counter}")
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

            env.reset()
            max_op_counts = test_dataset.max_op_counts.copy()
            job_op_counter = {job: 0 for job in range(env.n_jobs)}

    if valid_solution_count != num_predictions:
        print(f"Warning: Only {valid_solution_count} valid solutions were collected.")

    return all_predictions

def main():
    datasets = [
        'fjsspdataset/HurinkEdata1.fjs',
        # 'fjsspdataset/HurinkRdata5.fjs',
        # 'fjsspdataset/HurinkRdata6.fjs',
    ]
    num_episodes_per_dataset = 20  # 각 데이터셋당 에피소드 수 설정

    task_agent, machine_agent, state_size, action_size = train_individual_models(datasets, num_episodes_per_dataset)

    test_dataset = RLDataset_FJSSP('fjsspdataset/HurinkEdata1.fjs')

    logging.info("Training completed for individual models.")
    print("Training completed for individual models.")

    logging.info("Creating environment object.")
    print("Creating environment object.")

    process_times = [[(m, t) for m, t in ops] for job in test_dataset.op_data for ops in job]
    machine_sequence = [[[(m, t) for m, t in ops] for ops in job] for job in test_dataset.op_data]

    env = JobShopEnv_FJSSP(process_times, machine_sequence, solutions=None)

    logging.info("Starting prediction process.")
    print("Starting prediction process.")
    all_predictions = predict(task_agent, machine_agent, env, test_dataset, num_predictions=20, max_steps=100000000)

    logging.info(f"Predicted Solutions: {all_predictions}")
    print("Predicted Solutions:", all_predictions)

    best_solution = None
    best_makespan = float('inf')
    for solution in all_predictions:
        makespan = calculate_makespan(solution, test_dataset)
        if makespan < best_makespan:
            best_makespan = makespan
            best_solution = solution

    logging.info(f"Best Solution: {best_solution}")
    print("Best Solution:", best_solution)

    ga_initial_population = [job * test_dataset.n_machine + op for job, op, machine, _ in best_solution]
    logging.info(f"GA Initial Population: {ga_initial_population}")
    print("GA Initial Population:", ga_initial_population)

    draw_gantt_chart(best_solution, test_dataset)

def calculate_makespan(solution, dataset):
    # 각 작업의 시작 시간을 저장할 딕셔너리 초기화
    job_start_times = {job: 0 for job in range(dataset.n_job)}
    # 각 기계의 사용 가능한 시간을 저장할 딕셔너리 초기화
    machine_avail_times = {machine: 0 for machine in range(dataset.n_machine)}
    # makespan을 저장할 변수 초기화
    makespan = 0

    # 솔루션의 각 작업에 대해 시작 및 종료 시간을 계산
    for job, op, machine, duration in solution:
        start_time = max(job_start_times[job], machine_avail_times[machine])
        end_time = start_time + duration
        # 작업의 종료 시간을 업데이트
        job_start_times[job] = end_time
        # 기계의 사용 가능 시간을 업데이트
        machine_avail_times[machine] = end_time
        # 현재 작업의 종료 시간과 비교하여 makespan 업데이트
        makespan = max(makespan, end_time)

    # 최종 makespan 반환
    return makespan

def generate_colors(n):
    colors = plt.colormaps['tab20'](range(n))
    return [mcolors.rgb2hex(c) for c in colors]

def color(row, color_map):
    return color_map[row['Job']]

def draw_gantt_chart(predictions, dataset):
    job_start_times = {job: 0 for job in range(dataset.n_job)}
    machine_avail_times = {machine: 0 for machine in range(dataset.n_machine)}
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
    ax.set_yticks(np.arange(0, dataset.n_machine + 1, 1))
    ax.set_yticklabels(np.arange(0, dataset.n_machine + 1, 1))

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