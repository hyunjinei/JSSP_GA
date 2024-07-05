import os
import pandas as pd
from RLDataset_FJSSP import RLDataset_FJSSP
# from JobShopEnv_FJSSP_machine import JobShopEnv_FJSSP
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
import math

logging.basicConfig(filename='training_progress.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

class DuelingDQN(nn.Module):
    def __init__(self, max_state_size, action_size, atom_size=101, v_min=-100, v_max=100):
        super(DuelingDQN, self).__init__()
        self.action_size = action_size
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max

        self.feature = nn.Sequential(
            NoisyLinear(max_state_size, 128),
            nn.Tanh(),
            NoisyLinear(128, 128),
            nn.Tanh()
        )
        
        self.advantage = nn.Sequential(
            NoisyLinear(128, 128),
            nn.Tanh(),
            NoisyLinear(128, action_size * atom_size)
        )
        
        self.value = nn.Sequential(
            NoisyLinear(128, 128),
            nn.Tanh(),
            NoisyLinear(128, atom_size)
        )

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        feature = self.feature(x)
        advantage = self.advantage(feature).view(-1, self.action_size, self.atom_size)
        value = self.value(feature).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(q_atoms, dim=-1)

    def reset_noise(self):
        for layer in self.feature:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.value:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

class DQNAgent:
    def __init__(self, max_state_size, action_size, update_target_frequency=32, replay_start_size=1000):
        self.state_size = max_state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000000)
        self.gamma = 0.95 # reward 의 decay rate
        self.epsilon = 0.8  # 탐험(exploration) 확률 초기값
        self.epsilon_min = 0.01  # 탐험 확률의 최소값
        self.epsilon_decay = 0.995  # 탐험 확률 감소율
        self.learning_rate = 0.0001  # 학습률 0.0001
        self.model = DuelingDQN(max_state_size, action_size)
        self.target_model = DuelingDQN(max_state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = F.smooth_l1_loss
        self.v_min = -100 # -10
        self.v_max = 100 # 10
        self.atom_size = 101 # 51
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size)
        self.update_target_frequency = update_target_frequency
        self.replay_start_size = replay_start_size
        self.update_counter = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        print("Updated target network.")

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    # def act(self, state, mask=None):
    #     state = torch.FloatTensor(state).unsqueeze(0)
    #     if mask is not None:
    #         mask = torch.FloatTensor(mask).unsqueeze(0)
    #     q_dist = self.model(state, mask).detach()
    #     q_value = (q_dist * self.support.unsqueeze(0).unsqueeze(0)).sum(2)
        
    #     if np.random.rand() <= self.epsilon:
    #         return np.random.choice(self.action_size)
    #     return q_value.max(1)[1].item()

    def act(self, state, mask=None):
        state = torch.FloatTensor(state).unsqueeze(0)
        if mask is not None:
            mask = torch.FloatTensor(mask).unsqueeze(0)
        q_dist = self.model(state, mask).detach()
        q_value = (q_dist * self.support.unsqueeze(0).unsqueeze(0)).sum(2)
        
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            # 확률적으로 행동 선택
            probs = F.softmax(q_value, dim=1).squeeze().numpy()
            return np.random.choice(self.action_size, p=probs)

    def replay(self, batch_size):
        if len(self.memory) < self.replay_start_size:
            return

        batch, idxs, is_weights = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        is_weights = torch.FloatTensor(is_weights)

        curr_q_dist = self.model(states)
        next_q_dist = self.target_model(next_states)
        
        curr_q = curr_q_dist[range(batch_size), actions]
        next_q = next_q_dist.mean(2).max(1)[0]
        expected_q = rewards + (1 - dones) * self.gamma * next_q

        loss = (curr_q - expected_q.unsqueeze(1)).pow(2) * is_weights.unsqueeze(1)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.update_priorities(idxs, loss.detach().cpu().numpy())

        if self.update_counter % self.update_target_frequency == 0:
            self.update_target_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.model.reset_noise()
        self.target_model.reset_noise()

# PrioritizedReplayBuffer 클래스 추가
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.epsilon = epsilon

    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        self.beta = np.min([1., self.beta + self.beta_increment])
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = list(zip(*samples))
        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon

    def __len__(self):
        return len(self.buffer)

# TaskAgent와 MachineAgent 클래스 수정
class TaskAgent(DQNAgent):
    def __init__(self, max_state_size, action_size, update_target_frequency=32, replay_start_size=3000):
        super(TaskAgent, self).__init__(max_state_size, action_size, update_target_frequency, replay_start_size)
        self.memory = PrioritizedReplayBuffer(capacity=20000000)

class MachineAgent(DQNAgent):
    def __init__(self, max_state_size, action_size, update_target_frequency=32, replay_start_size=3000):
        super(MachineAgent, self).__init__(max_state_size, action_size, update_target_frequency, replay_start_size)
        self.memory = PrioritizedReplayBuffer(capacity=20000000)

def pad_state_action(state, action, max_state_size=1000, pad_value=-2):
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

def train_individual_models(datasets, num_episodes_per_dataset):
    max_state_size = 20 * 10 * 5
    action_size = 20 * 10

    task_agent = TaskAgent(max_state_size, action_size, update_target_frequency=320, replay_start_size=1000)
    machine_agent = MachineAgent(max_state_size, action_size, update_target_frequency=320, replay_start_size=1000)

    for i, dataset in enumerate(datasets):
        print(f"Starting training for dataset {dataset} ({i+1}/{len(datasets)})")

        data = RLDataset_FJSSP(dataset)

        process_times = [[(m, t) for m, t in ops] for job in data.op_data for ops in job]
        machine_sequence = [[[(m, t) for m, t in ops] for ops in job] for job in data.op_data]

        for job in range(len(machine_sequence)):
            print(f"Job {job}: {machine_sequence[job]}")

        env = JobShopEnv_FJSSP(process_times, machine_sequence, solutions=None)

        state_size = env.n_jobs * env.n_machines * 3
        action_size = env.n_jobs * env.n_machines

        for episode in range(num_episodes_per_dataset):
            state = env.reset()
            state, mask = pad_state_action(np.concatenate((
                np.array([state[0]]), 
                state[1], 
                state[2], 
                state[3],
                state[4],
                # state[5]
            )), [])
            max_op_counts = [len(ops) for ops in machine_sequence]
            job_op_counter = [0] * env.n_jobs

            done = False
            episode_reward = 0
            while not done:
                task_action = task_agent.act(state, mask)
                selected_job, selected_op = divmod(task_action, env.n_machines)

                if selected_job >= env.n_jobs or selected_op >= max_op_counts[selected_job] or job_op_counter[selected_job] != selected_op:
                    continue

                machine_action = machine_agent.act(state, mask)
                selected_machine = machine_action % env.n_machines

                machine_options = env.machine_sequence[selected_job][selected_op]
                valid_machines = [m for m, t in machine_options]
                if selected_machine not in valid_machines:
                    continue

                next_state, done, step_reward = env.step(selected_job, selected_op, selected_machine)
                next_state, next_mask = pad_state_action(np.concatenate((
                    np.array([next_state[0]]), 
                    next_state[1], 
                    next_state[2], 
                    next_state[3],
                    next_state[4],
                    # next_state[5]
                )), [])

                task_agent.memory.add(state, selected_job * env.n_machines + selected_op, step_reward, next_state, done)
                machine_agent.memory.add(state, selected_op * env.n_jobs + selected_job, step_reward, next_state, done)
                state, mask = next_state, next_mask

                job_op_counter[selected_job] += 1
                if job_op_counter[selected_job] > max_op_counts[selected_job]:
                    job_op_counter[selected_job] = max_op_counts[selected_job]

                episode_reward += step_reward * 10

                if done or all(count >= max_op_counts[j] for j, count in enumerate(job_op_counter)):
                    reward_task, reward_machine = env.calculate_episode_rewards()
                    print(f"Training: Episode {episode+1} processed. Reward Task: {reward_task}, Reward Machine: {reward_machine}, Step Reward: {episode_reward}")
                    break

            if len(task_agent.memory) >= task_agent.replay_start_size:
                for _ in range(10):
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

    return task_agent, machine_agent, max_state_size, action_size

def predict(task_agent, machine_agent, env, test_dataset, num_predictions=1, max_steps=100000000):
    valid_solution_count = 0
    all_predictions = []
    step_count = 0

    while valid_solution_count < num_predictions and step_count < max_steps:
        state = env.reset()
        max_op_counts = test_dataset.max_op_counts.copy()
        job_op_counter = {job: 0 for job in range(env.n_jobs)}

        state = np.concatenate((
            np.array([state[0]]),
            state[1],
            state[2],
            state[3],  # 추가된 기계 평균 이용률
            state[4],  # 추가된 작업 큐 길이
            # state[5]  # 추가된 작업 진행률
            # state[6]   # 추가된 작업 남은 시간
        ))
        state, mask = pad_state_action(state, [], max_state_size=1000)

        solution = []
        done = False
        episode_reward = 0

        print(f"Initial max_op_counts: {max_op_counts}")

        while not done and step_count < max_steps:
            start_time = time.time()
            
            task_action = task_agent.act(state, mask)
            job, op = divmod(task_action, env.n_machines)

            if job >= env.n_jobs or op >= len(env.machine_sequence[job]) or job_op_counter[job] != op:
                continue

            machine_action = machine_agent.act(state, mask)
            machine = machine_action % env.n_machines

            machine_options = env.machine_sequence[job][op]
            machine_duration = next((t for m, t in machine_options if m == machine), None)
            if machine_duration is None:
                continue

            solution.append((job, op, machine, machine_duration))
            job_op_counter[job] += 1
            step_count += 1

            if job_op_counter[job] > max_op_counts[job]:
                max_op_counts[job] = -1
                print(f"Job {job} completed. Updated max_op_counts: {max_op_counts}")

            if step_count % 100 == 0:
                print(f"Predicting: Step {step_count}, Current Solution: {solution[-5:]}")
                gc.collect()

            next_state, done, step_reward = env.step(job, op, machine)
            state = np.concatenate((
                np.array([next_state[0]]),
                next_state[1],
                next_state[2],
                next_state[3],
                next_state[4],
                # next_state[5]
                # next_state[6]
            ))
                
            state, mask = pad_state_action(state, [], max_state_size=1000)
            end_time = time.time()
            print(f"Step {step_count}, job_op_counter: {job_op_counter}")
            print(f"Step idle time reward: {step_reward}")
            episode_reward += step_reward * 10

        if done:
            reward_task, reward_machine = env.calculate_episode_rewards()
            print(f"Prediction: Episode {valid_solution_count+1}/{num_predictions} processed. "
                  f"Reward Task: {reward_task}, Reward Machine: {reward_machine}, Episode Reward: {episode_reward}")
            valid_solution_count += 1
            all_predictions.append(solution)
            gc.collect()

            env.reset()
            max_op_counts = test_dataset.max_op_counts.copy()
            job_op_counter = {job: 0 for job in range(env.n_jobs)}

    if valid_solution_count != num_predictions:
        print(f"Warning: Only {valid_solution_count} valid solutions were collected.")

    return all_predictions
# main 함수
def main():
    # 데이터셋 경로를 지정. 다양한 데이터셋을 사용하도록 수정
    datasets = [
        'fjsspdataset/HurinkRdata5.fjs',
        # 'fjsspdataset/HurinkRdata5.fjs',
        # 'fjsspdataset/HurinkRdata6.fjs',
    ]
    num_episodes_per_dataset = 20  # 각 데이터셋당 에피소드 수 설정

    # 개별 모델을 훈련시키고, task_agent와 machine_agent를 반환받음
    task_agent, machine_agent, max_state_size, action_size = train_individual_models(datasets, num_episodes_per_dataset)

    # 테스트 데이터셋 로드. 이 예제에서는 하나의 테스트 데이터셋을 사용
    test_dataset = RLDataset_FJSSP('fjsspdataset/HurinkRdata5.fjs')

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

    task_agent.model.reset_noise()
    machine_agent.model.reset_noise()
    # 예측을 시작하여 지정된 수(num_predictions)만큼의 솔루션을 생성
    all_predictions = predict(task_agent, machine_agent, env, test_dataset, num_predictions=20, max_steps=100000000)

    logging.info(f"Predicted Solutions: {all_predictions}")
    print("Predicted Solutions:", all_predictions)

    # 가장 좋은 솔루션을 찾기 위해 초기화
    best_solution = None
    best_makespan = float('inf')
    # 모든 예측된 솔루션에 대해 makespan을 계산하고, 가장 작은 makespan을 가진 솔루션을 선택
    for solution in all_predictions:
        makespan = calculate_makespan(solution, test_dataset)  # FJSSP에 맞는 makespan 계산 함수로 수정
        if makespan < best_makespan:
            best_makespan = makespan
            best_solution = solution

    logging.info(f"Best Solution: {best_solution}")
    print("Best Solution:", best_solution)

    # GA 초기 인구 생성: best_solution에서 작업과 작업 단계를 이용해 초기 인구를 생성
    ga_initial_population = [job * test_dataset.n_machine + op for job, op, machine, _ in best_solution]
    logging.info(f"GA Initial Population: {ga_initial_population}")
    print("GA Initial Population:", ga_initial_population)

    # 간트 차트를 그려 best_solution을 시각화
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
       

