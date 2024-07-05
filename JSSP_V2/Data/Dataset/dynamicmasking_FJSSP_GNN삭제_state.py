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

logging.basicConfig(filename='training_progress.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class DQN(nn.Module):
    def __init__(self, max_state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(max_state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, max_state_size, action_size, update_target_frequency=32, replay_start_size=1000):
        self.state_size = max_state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000000)
        self.gamma = 0.75 # 할인율 (미래 보상의 현재 가치)
        self.epsilon = 0.8  # 탐험(exploration) 확률 초기값
        self.epsilon_min = 0.001  # 탐험 확률의 최소값
        self.epsilon_decay = 0.995  # 탐험 확률 감소율
        self.learning_rate = 0.0001  # 학습률 0.0001
        self.model = DQN(max_state_size, action_size)
        self.target_model = DQN(max_state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = F.smooth_l1_loss
        self.update_target_frequency = update_target_frequency
        self.replay_start_size = replay_start_size
        self.update_counter = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        print("Updated target network.")

    def remember(self, state, action, reward, next_state, done):
        # state와 next_state를 1차원 배열로 변환
        state = np.array(state).flatten()
        next_state = np.array(next_state).flatten()
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, mask=None):
        state = torch.FloatTensor(state).unsqueeze(0)
        if mask is not None:
            mask = torch.FloatTensor(mask).unsqueeze(0)
        act_values = self.model(state, mask).detach().numpy()[0]
        
        if np.random.rand() <= self.epsilon:
            # ε 확률로 랜덤 행동 선택 (탐험)
            return random.randrange(self.action_size)
        else:
            # 1-ε 확률로 최적 행동 선택 (탐사)
            return np.argmax(act_values)

    def replay(self, batch_size):
        if len(self.memory) < self.replay_start_size:
            return  

        minibatch = random.sample(self.memory, batch_size)
        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            state, mask = pad_state_action(state, [])
            next_state, next_mask = pad_state_action(next_state, [])
            state = torch.FloatTensor(state).unsqueeze(0)
            mask = torch.FloatTensor(mask).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            next_mask = torch.FloatTensor(next_mask).unsqueeze(0)

                # 현재 상태에서의 예측값 계산
            current_q_values = self.model(state, mask)
            current_q_value = current_q_values[0][action]

            # 타겟 네트워크를 사용하여 타겟 값 계산
            with torch.no_grad():
                next_q_values = self.target_model(next_state, next_mask)
                max_next_q_value = next_q_values[0].max()
                target = reward + (self.gamma * max_next_q_value * (1 - done))

            # 손실 계산
            loss = self.criterion(current_q_value, target)  # Huber 손실 또는 MSE 손실 함수 사용
            
            self.optimizer.zero_grad()  # 옵티마이저의 그래디언트 초기화
            loss.backward()  # 역전파
            self.optimizer.step()  # 모델 파라미터 업데이트
            
            if idx % 31 == 0:  
                # 매 스텝마다 로그를 남기고 메모리 관리
                logging.info(f"Replay: Step {idx+1}/{batch_size} processed. Loss: {loss.item()}")
                print(f"Replay: Step {idx+1}/{batch_size} processed. Loss: {loss.item()}")
                gc.collect()

            self.update_counter += 1  # 업데이트 카운터 증가

        if self.update_counter % self.update_target_frequency == 0:
            # 일정 횟수마다 타겟 네트워크 업데이트
            self.update_target_model()
            print("Updated target network after replay.") 

        if self.epsilon > self.epsilon_min:
            # 탐험 확률 감소
            self.epsilon *= self.epsilon_decay



class TaskAgent(DQNAgent):
    def __init__(self, max_state_size, action_size, update_target_frequency=32, replay_start_size=3000):
        super(TaskAgent, self).__init__(max_state_size, action_size, update_target_frequency, replay_start_size)

    def act(self, state, mask=None, env=None):
        return super(TaskAgent, self).act(state, mask)

class MachineAgent(DQNAgent):
    def __init__(self, max_state_size, action_size, update_target_frequency=32, replay_start_size=3000):
        super(MachineAgent, self).__init__(max_state_size, action_size, update_target_frequency, replay_start_size)

    def act(self, state, mask=None, env=None):
        return super(MachineAgent, self).act(state, mask)

def pad_state_action(state, action, max_state_size=1000, pad_value=-2):
    state = np.array(state).flatten()  # state를 1차원 배열로 변환
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

    task_agent = TaskAgent(max_state_size, action_size, update_target_frequency=128, replay_start_size=100)
    machine_agent = MachineAgent(max_state_size, action_size, update_target_frequency=128, replay_start_size=100)

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
            task_state = np.concatenate((
                np.array([state[0]]), 
                state[1],
                state[4],
                state[5],
            ))
            machine_state = np.concatenate((
                np.array([state[0]]), 
                state[2],
                state[3],
                state[5],
            ))
            task_state, task_mask = pad_state_action(task_state, [])
            machine_state, machine_mask = pad_state_action(machine_state, [])

            max_op_counts = [len(ops) for ops in machine_sequence]
            job_op_counter = [0] * env.n_jobs

            done = False
            episode_reward = 0
            step_count = 0
            while not done:
                task_action = task_agent.act(task_state, task_mask, env)
                selected_job, selected_op = divmod(task_action, env.n_machines)

                if selected_job >= env.n_jobs or selected_op >= max_op_counts[selected_job] or job_op_counter[selected_job] != selected_op:
                    continue

                machine_action = machine_agent.act(machine_state, machine_mask, env)
                selected_machine = machine_action % env.n_machines

                machine_options = env.machine_sequence[selected_job][selected_op]
                valid_machines = [m for m, t in machine_options]
                if selected_machine not in valid_machines:
                    continue

                next_state, done, step_reward = env.step(selected_job, selected_op, selected_machine)
                next_task_state = np.concatenate((
                    np.array([next_state[0]]), 
                    next_state[1],
                    next_state[4],
                    next_state[5],
                ))
                next_machine_state = np.concatenate((
                    np.array([next_state[0]]), 
                    next_state[2],
                    next_state[3],
                    next_state[5],
                ))
                next_task_state, next_task_mask = pad_state_action(next_task_state, [])
                next_machine_state, next_machine_mask = pad_state_action(next_machine_state, [])

                task_agent.remember(task_state, selected_job * env.n_machines + selected_op, step_reward, next_task_state, done)
                machine_agent.remember(machine_state, selected_op * env.n_jobs + selected_job, step_reward, next_machine_state, done)
                task_state, task_mask = next_task_state, next_task_mask
                machine_state, machine_mask = next_machine_state, next_machine_mask

                job_op_counter[selected_job] += 1
                if job_op_counter[selected_job] > max_op_counts[selected_job]:
                    job_op_counter[selected_job] = max_op_counts[selected_job]

                episode_reward += step_reward

                step_count += 1
                if step_count % 30 == 0:  # 100 스텝마다 gc.collect() 호출
                    gc.collect()
                    print(f"Step {step_count} completed. Memory cleaned.")

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
                gc.collect()  # 각 에피소드가 끝날 때마다 gc.collect() 호출

        print(f"Completed training for dataset {dataset} ({i+1}/{len(datasets)})")
        gc.collect()  # 각 데이터셋 훈련이 끝날 때마다 gc.collect() 호출

    return task_agent, machine_agent, max_state_size, action_size

def predict(task_agent, machine_agent, env, test_dataset, num_predictions=1, max_steps=100000000):
    valid_solution_count = 0
    all_predictions = []
    step_count = 0

    while valid_solution_count < num_predictions and step_count < max_steps:
        state = env.reset()
        max_op_counts = test_dataset.max_op_counts.copy()
        job_op_counter = {job: 0 for job in range(env.n_jobs)}

        task_state = np.concatenate((
            np.array([state[0]]),
            state[1],
            state[4],
            state[5],
        ))
        machine_state = np.concatenate((
            np.array([state[0]]),
            state[2],
            state[3],
            state[5],
        ))
        task_state, task_mask = pad_state_action(task_state, [], max_state_size=1000)
        machine_state, machine_mask = pad_state_action(machine_state, [], max_state_size=1000)

        # state, mask = pad_state_action(state, [], max_state_size=1000)

        solution = []
        done = False
        episode_reward = 0

        print(f"Initial max_op_counts: {max_op_counts}")

        while not done and step_count < max_steps:
            task_action = task_agent.act(task_state, task_mask, env)
            job, op = divmod(task_action, env.n_machines)

            if job >= env.n_jobs or op >= len(env.machine_sequence[job]) or job_op_counter[job] != op:
                continue

            machine_action = machine_agent.act(machine_state, machine_mask, env)
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

            next_task_state = np.concatenate((
                np.array([next_state[0]]),
                next_state[1],
                next_state[4],
                next_state[5],
            ))
            next_machine_state = np.concatenate((
                np.array([next_state[0]]),
                next_state[2],
                next_state[3],
                next_state[5],
            ))
            next_task_state, next_task_mask = pad_state_action(next_task_state, [], max_state_size=1000)
            next_machine_state, next_machine_mask = pad_state_action(next_machine_state, [], max_state_size=1000)
            
            task_state, task_mask = next_task_state, next_task_mask
            machine_state, machine_mask = next_machine_state, next_machine_mask
 
            # state, mask = pad_state_action(state, [], max_state_size=1000)
            print(f"Step {step_count}, job_op_counter: {job_op_counter}")
            print(f"Step reward: {step_reward}")

        if done:
            reward_task, reward_machine = env.calculate_episode_rewards()
            reward_task += episode_reward * 1
            reward_machine += episode_reward * 1
            print(f"Prediction: Episode {valid_solution_count+1}/{num_predictions} processed. "
                  f"Reward Task: {reward_task}, Reward Machine: {reward_machine},Episode every step Reward: {episode_reward}")
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
        'fjsspdataset/HurinkEdata1.fjs',
        # 'fjsspdataset/HurinkRdata5.fjs',
        # 'fjsspdataset/HurinkRdata6.fjs',
    ]
    num_episodes_per_dataset = 20  # 각 데이터셋당 에피소드 수 설정

    # 개별 모델을 훈련시키고, task_agent와 machine_agent를 반환받음
    task_agent, machine_agent, max_state_size, action_size = train_individual_models(datasets, num_episodes_per_dataset)

    # 테스트 데이터셋 로드. 이 예제에서는 하나의 테스트 데이터셋을 사용
    test_dataset = RLDataset_FJSSP('fjsspdataset/HurinkEdata1.fjs')

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
       

