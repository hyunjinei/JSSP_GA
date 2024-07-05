import os
import pandas as pd
from RLDataset_FJSSP import RLDataset_FJSSP
from JobShopEnv_FJSSP_machine_copy import JobShopEnv_FJSSP
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
    def __init__(self, max_state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(max_state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, action_size)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    def __init__(self, max_state_size, action_size, update_target_frequency=32, replay_start_size=1000, batch_size=64):
        self.state_size = max_state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.99
        self.epsilon = 0.99
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.model = DQN(max_state_size, action_size)
        self.target_model = DQN(max_state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = F.smooth_l1_loss
        self.update_target_frequency = update_target_frequency
        self.replay_start_size = replay_start_size
        self.update_counter = 0
        self.batch_size = batch_size

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        print("Updated target network.")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, mask=None):
        state = torch.FloatTensor(state).unsqueeze(0)
        if mask is not None:
            mask = torch.FloatTensor(mask).unsqueeze(0)
        act_values = self.model(state, mask).detach().numpy()[0]
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(act_values)

    def replay(self):
        if len(self.memory) < self.replay_start_size:
            return  

        minibatch = random.sample(self.memory, self.batch_size)
        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            state, mask = pad_state_action(state, [])
            next_state, next_mask = pad_state_action(next_state, [])
            state = torch.FloatTensor(state).unsqueeze(0)
            mask = torch.FloatTensor(mask).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            next_mask = torch.FloatTensor(next_mask).unsqueeze(0)

            current_q_values = self.model(state, mask)
            current_q_value = current_q_values[0][action]

            with torch.no_grad():
                next_q_values = self.target_model(next_state, next_mask)
                max_next_q_value = next_q_values[0].max()
                target = reward + (self.gamma * max_next_q_value * (1 - done))

            loss = self.criterion(current_q_value, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if idx % 31 == 0:
                logging.info(f"Replay: Step {idx+1}/{self.batch_size} processed. Loss: {loss.item()}")
                print(f"Replay: Step {idx+1}/{self.batch_size} processed. Loss: {loss.item()}")
                gc.collect()

            self.update_counter += 1

        if self.update_counter % self.update_target_frequency == 0:
            self.update_target_model()
            print("Updated target network after replay.") 

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class TaskAgent(DQNAgent):
    def __init__(self, max_state_size, action_size, update_target_frequency=32, replay_start_size=300):
        super(TaskAgent, self).__init__(max_state_size, action_size, update_target_frequency, replay_start_size)

    def act(self, state, mask=None, env=None):
        return super(TaskAgent, self).act(state, mask)

class MachineAgent(DQNAgent):
    def __init__(self, max_state_size, action_size, update_target_frequency=32, replay_start_size=300):
        super(MachineAgent, self).__init__(max_state_size, action_size, update_target_frequency, replay_start_size)

    def act(self, state, mask=None, env=None):
        return super(MachineAgent, self).act(state, mask)

def pad_state_action(state, action, max_state_size=1200, pad_value=-2):
    if len(state) > max_state_size:
        state = state[:max_state_size]
    
    max_action_size = 20 * 10
    padded_state = np.full(max_state_size, pad_value)
    padded_state[:len(state)] = state
    mask = np.zeros(max_state_size)
    mask[:len(state)] = 1
    padded_action = np.full(max_action_size, pad_value)
    padded_action[:len(action)] = action
    return padded_state, mask

def flatten_operation_view(operation_view):
    flattened = []
    for op_list in operation_view:
        for job_ops in op_list:
            for machine_time in job_ops:
                flattened.extend(machine_time)
    return np.array(flattened, dtype=float)

def flatten_machine_view(machine_view):
    return np.array([item for machine_ops in machine_view for item in machine_ops], dtype=float)

def train_individual_models(datasets, num_episodes_per_dataset):
    max_state_size = 20 * 10 * 6
    action_size = 20 * 10

    task_agent = TaskAgent(max_state_size, action_size, update_target_frequency=128, replay_start_size=500)
    machine_agent = MachineAgent(max_state_size, action_size, update_target_frequency=128, replay_start_size=500)

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
                # Task Agent (Operation 관점)
                task_state = np.concatenate((
                    np.array([state['current_time']]),
                    state['job_completion'],
                    state['machine_available_time'],
                    state['job_progress'],
                    state['remaining_job_time'],
                    np.array([status for job_status in state['job_op_status'] for status in job_status])
                ))
                task_state, task_mask = pad_state_action(task_state, [])
                task_action = task_agent.act(task_state, task_mask, env)
                selected_job, selected_op = divmod(task_action, env.n_machines)

                # Machine Agent (Machine 관점)
                machine_state = np.concatenate((
                    np.array([state['current_time']]),
                    state['job_completion'],
                    state['machine_available_time'],
                    state['machine_utilization'],
                    state['job_queue_length'],
                    np.array([status for job_status in state['job_op_status'] for status in job_status])                    
                ))
                machine_state, machine_mask = pad_state_action(machine_state, [])
                machine_action = machine_agent.act(machine_state, machine_mask, env)
                selected_machine = machine_action % env.n_machines

                # 유효한 액션 확인 및 제약 조건 적용
                valid_actions = env.get_valid_actions()
                if (selected_job, selected_op, selected_machine) not in valid_actions:
                    continue

                # 기계 선택 제약 조건 적용
                if env.assigned_machines[selected_machine] == 0:
                    continue

                next_state, done, step_reward = env.step(selected_job, selected_op, selected_machine, episode)

                # Task Agent 메모리 업데이트
                next_task_state = np.concatenate((
                    np.array([next_state['current_time']]),
                    next_state['job_completion'],
                    next_state['machine_available_time'],
                    next_state['job_progress'],
                    next_state['remaining_job_time'],
                    np.array([status for job_status in next_state['job_op_status'] for status in job_status])
                ))
                next_task_state, _ = pad_state_action(next_task_state, [])
                task_agent.remember(task_state, task_action, step_reward, next_task_state, done)

                # Machine Agent 메모리 업데이트
                next_machine_state = np.concatenate((
                    np.array([next_state['current_time']]),
                    next_state['job_completion'],
                    next_state['machine_available_time'],
                    next_state['machine_utilization'],
                    next_state['job_queue_length'],
                    np.array([status for job_status in next_state['job_op_status'] for status in job_status])
                ))
                next_machine_state, _ = pad_state_action(next_machine_state, [])
                machine_agent.remember(machine_state, machine_action, step_reward, next_machine_state, done)

                state = next_state
                episode_reward += step_reward

                step_count += 1
                if step_count % 100 == 0:
                    gc.collect()
                    print(f"Step {step_count} completed. Memory cleaned.")

                if done:
                    reward_task, reward_machine = env.calculate_episode_rewards()
                    reward_task += episode_reward
                    reward_machine += episode_reward
                    print(f"Training: Episode {episode+1} processed. Reward Task: {reward_task}, Reward Machine: {reward_machine}, episode reward every step: {episode_reward}")
                    break

            # 에이전트 학습
            if len(task_agent.memory) >= task_agent.replay_start_size:
                for _ in range(2):
                    task_agent.replay(64)
                    machine_agent.replay(64)

            # 타겟 네트워크 업데이트
            if task_agent.update_counter % task_agent.update_target_frequency == 0:
                task_agent.update_target_model()
                print("Updated target network for task agent during training.")
            
            if machine_agent.update_counter % machine_agent.update_target_frequency == 0:
                machine_agent.update_target_model()
                print("Updated target network for machine agent during training.")

            if episode % 1 == 0:
                print(f"Episode {episode}/{num_episodes_per_dataset} completed for dataset {dataset}")
                gc.collect()

        print(f"Completed training for dataset {dataset} ({i+1}/{len(datasets)})")
        gc.collect()

    return task_agent, machine_agent, max_state_size, action_size

def predict(task_agent, machine_agent, env, test_dataset, num_predictions=1, max_steps=100000000):
    valid_solution_count = 0
    all_predictions = []
    step_count = 0

    while valid_solution_count < num_predictions and step_count < max_steps:
        state = env.reset()
        solution = []
        done = False
        episode_reward = 0

        while not done and step_count < max_steps:
            # Task Agent (Operation 관점)
            task_state = np.concatenate((
                np.array([state['current_time']]),
                state['job_completion'],
                state['machine_available_time'],
                state['job_progress'],
                state['remaining_job_time'],
                np.array([status for job_status in state['job_op_status'] for status in job_status])
            ))
            task_state, task_mask = pad_state_action(task_state, [], max_state_size=1200)
            task_action = task_agent.act(task_state, task_mask)
            job, op = divmod(task_action, env.n_machines)

            # Machine Agent (Machine 관점)
            machine_state = np.concatenate((
                np.array([state['current_time']]),
                state['job_completion'],
                state['machine_available_time'],
                state['machine_utilization'],
                state['job_queue_length'],
                np.array([status for job_status in state['job_op_status'] for status in job_status])
            ))
            machine_state, machine_mask = pad_state_action(machine_state, [], max_state_size=1200)
            machine_action = machine_agent.act(machine_state, machine_mask)
            selected_machine = machine_action % env.n_machines

            # 유효한 액션 확인 및 제약 조건 적용
            valid_actions = env.get_valid_actions()
            if (job, op, selected_machine) not in valid_actions:
                continue

            # 기계 선택 제약 조건 적용
            if env.assigned_machines[selected_machine] == 0:
                available_machines = [m for m, _ in env.machine_sequence[job][op] if env.assigned_machines[m] == 1]
                if available_machines:
                    selected_machine = random.choice(available_machines)
                else:
                    env.assigned_machines = [1] * env.n_machines
                    selected_machine = random.choice([m for m, _ in env.machine_sequence[job][op]])

            next_state, done, step_reward = env.step(job, op, selected_machine, 0)  # 0을 에피소드로 사용
            duration = next(t for m, t in env.machine_sequence[job][op] if m == selected_machine)
            solution.append((job, op, selected_machine, duration))
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




# main 함수
def main():
    # 데이터셋 경로를 지정. 다양한 데이터셋을 사용하도록 수정
    datasets = [
        'fjsspdataset/Fattahi10.fjs',
        # 'fjsspdataset/HurinkRdata10.fjs',
        # 'fjsspdataset/HurinkRdata11.fjs',
        # 'fjsspdataset/HurinkRdata12.fjs',
    ]
    num_episodes_per_dataset = 10

    # 개별 모델을 훈련시키고, task_agent와 machine_agent를 반환받음
    task_agent, machine_agent, max_state_size, action_size = train_individual_models(datasets, num_episodes_per_dataset)

    # 테스트 데이터셋 로드
    test_dataset = RLDataset_FJSSP('fjsspdataset/Fattahi10.fjs')

    logging.info("Training completed for individual models.")
    print("Training completed for individual models.")

    logging.info("Creating environment object.")
    print("Creating environment object.")

    # 환경 생성 시 FJSSP 환경 클래스를 사용하여 생성
    process_times = [[(m, t) for m, t in ops] for job in test_dataset.op_data for ops in job]
    machine_sequence = [[[(m, t) for m, t in ops] for ops in job] for job in test_dataset.op_data]

    # 환경 객체 생성
    env = JobShopEnv_FJSSP(process_times, machine_sequence, solutions=None)

    logging.info("Starting prediction process.")
    print("Starting prediction process.")
    # 예측을 시작하여 지정된 수(num_predictions)만큼의 솔루션을 생성
    all_predictions = predict(task_agent, machine_agent, env, test_dataset, num_predictions=10, max_steps=100000000)

    logging.info(f"Predicted Solutions: {all_predictions}")
    print("Predicted Solutions:", all_predictions)

    # 가장 좋은 솔루션을 찾기 위해 초기화
    best_solution = None
    best_makespan = float('inf')
    # 모든 예측된 솔루션에 대해 makespan을 계산하고, 가장 작은 makespan을 가진 솔루션을 선택
    for solution in all_predictions:
        makespan = calculate_makespan(solution, env)
        if makespan < best_makespan:
            best_makespan = makespan
            best_solution = solution

    logging.info(f"Best Solution: {best_solution}")
    print("Best Solution:", best_solution)

    # GA 초기 인구 생성: best_solution에서 작업과 작업 단계를 이용해 초기 인구를 생성
    ga_initial_population = [job * env.n_machines + op for job, op, _, _ in best_solution]
    logging.info(f"GA Initial Population: {ga_initial_population}")
    print("GA Initial Population:", ga_initial_population)

    # 간트 차트를 그려 best_solution을 시각화
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