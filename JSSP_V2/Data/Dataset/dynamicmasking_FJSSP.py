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

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim_1,hidden_dim_2,hidden_dim_3,output_dim):
        # GNN 클래스의 생성자. 입력 차원, 은닉층 차원, 출력 차원을 인자로 받습니다.
        super(GNN, self).__init__()
        # 첫 번째 GCN 레이어를 정의합니다. 입력 차원에서 은닉층 차원으로 변환합니다.
        self.conv1 = GCNConv(input_dim, hidden_dim_1)
        # 두 번째 GCN 레이어를 정의합니다. 은닉층 차원에서 출력 차원으로 변환합니다.
        self.conv2 = GCNConv(hidden_dim_1, hidden_dim_2)
        self.conv3 = GCNConv(hidden_dim_2, hidden_dim_3)
        self.conv4 = GCNConv(hidden_dim_3, output_dim)


    def forward(self, data):
        # forward 함수는 모델의 순전파를 정의합니다. 입력은 데이터 객체입니다.
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # 첫 번째 GCN 레이어를 통과한 후 ReLU 활성화 함수를 적용합니다.
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)     
        x = self.conv3(x, edge_index)
        x = torch.relu(x)             
        # 두 번째 GCN 레이어를 통과합니다.
        x = self.conv4(x, edge_index)
        return x

def create_graph(env):
    # 환경(env) 객체로부터 그래프 데이터를 생성합니다.
    nodes = []
    # 작업 노드를 생성합니다. 각 노드는 [1, 0]으로 표시됩니다.
    for i in range(env.n_jobs):
        nodes.append([1, 0])
    # 기계 노드를 생성합니다. 각 노드는 [0, 1]으로 표시됩니다.
    for j in range(env.n_machines):
        nodes.append([0, 1])
    
    edges = []
    edge_features = []
    # 각 작업의 각 작업 단계에 대해 엣지와 엣지 특성을 생성합니다.
    for job in range(env.n_jobs):
        for op, (machine, duration) in enumerate(env.process_times[job]):
            # 엣지는 작업 노드에서 기계 노드로 연결됩니다.
            edges.append([job, env.n_jobs + machine])
            # 엣지 특성은 해당 작업 단계의 소요 시간입니다.
            edge_features.append([duration])
    
    # 엣지와 엣지 특성을 텐서로 변환합니다.
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    # 노드 특성을 텐서로 변환합니다.
    x = torch.tensor(nodes, dtype=torch.float)
    # 데이터 객체를 생성합니다.
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data


class DQN(nn.Module):
    def __init__(self, max_state_size, action_size):
        # DQN 클래스의 생성자. 최대 상태 크기와 행동 크기를 인자로 받습니다.
        super(DQN, self).__init__()
        # 첫 번째 완전 연결 층을 정의합니다. 입력 차원에서 128 차원으로 변환합니다.
        self.fc1 = nn.Linear(max_state_size, 128)
        # 두 번째 완전 연결 층을 정의합니다. 128 차원에서 128 차원으로 변환합니다.
        self.fc2 = nn.Linear(128, 128)
        # 세 번째 완전 연결 층을 정의합니다. 128 차원에서 행동 차원으로 변환합니다.
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x, mask=None):
        # forward 함수는 모델의 순전파를 정의합니다. 입력은 상태 벡터 x입니다.
        if mask is not None:
            # 마스크가 주어진 경우, 상태 벡터에 마스크를 적용합니다.
            x = x * mask
        # 첫 번째 완전 연결 층을 통과한 후 ReLU 활성화 함수를 적용합니다.
        x = torch.tanh(self.fc1(x))
        # 두 번째 완전 연결 층을 통과한 후 ReLU 활성화 함수를 적용합니다.
        x = torch.tanh(self.fc2(x))
        # 세 번째 완전 연결 층을 통과합니다.
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, max_state_size, action_size, update_target_frequency=32, replay_start_size=1000):
        # DQNAgent 클래스의 생성자. 상태 크기, 행동 크기, 타겟 네트워크 업데이트 빈도, 리플레이 시작 크기를 인자로 받습니다.
        self.state_size = max_state_size  # 최대 상태 크기 설정
        self.action_size = action_size  # 행동 크기 설정
        self.memory = deque(maxlen=20000000)  # 경험을 저장하기 위한 메모리 버퍼 (최대 크기 2000만)
        self.gamma = 0.95  # 할인율 (미래 보상의 현재 가치)
        self.epsilon = 1.0  # 탐험(exploration) 확률 초기값
        self.epsilon_min = 0.01  # 탐험 확률의 최소값
        self.epsilon_decay = 0.995  # 탐험 확률 감소율
        self.learning_rate = 0.0001  # 학습률
        self.model = DQN(max_state_size, action_size)  # DQN 모델 생성
        self.target_model = DQN(max_state_size, action_size)  # 타겟 네트워크 생성
        # self.target_model2 = DQN(max_state_size, action_size)  # 타겟 네트워크 복사본 생성
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # Adam 옵티마이저 사용
        self.criterion = F.smooth_l1_loss  # Huber 손실 함수 사용
        self.update_target_frequency = update_target_frequency  # 타겟 네트워크 업데이트 빈도
        self.replay_start_size = replay_start_size  # 리플레이 시작 크기
        self.update_counter = 0  # 업데이트 횟수 카운터

    def update_target_model(self):
        # 타겟 네트워크를 현재 네트워크의 가중치로 업데이트합니다.
        self.target_model.load_state_dict(self.model.state_dict())
        # self.target_model2.load_state_dict(self.model.state_dict())
        print("Updated target network.")  # 업데이트 완료 메시지 출력

    def remember(self, state, action, reward, next_state, done):
        # 경험을 메모리에 저장합니다.
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, mask=None):
        # 행동을 선택합니다. 탐험/탐사 전략을 사용합니다.
        state = torch.FloatTensor(state).unsqueeze(0)  # 상태를 텐서로 변환하고 배치 차원 추가
        if mask is not None:
            mask = torch.FloatTensor(mask).unsqueeze(0)  # 마스크를 텐서로 변환하고 배치 차원 추가
        act_values = self.model(state, mask).detach().numpy()[0]  # 네트워크를 통해 행동 가치 예측
        
        max_action = np.argmax(act_values)  # 가장 높은 가치를 가지는 행동 선택
        probabilities = np.ones(self.action_size) * (self.epsilon / self.action_size)  # 모든 행동에 대한 확률 초기화
        probabilities[max_action] += (1.0 - self.epsilon)  # 선택한 행동의 확률을 증가
        
        return np.random.choice(self.action_size, p=probabilities)  # 확률에 따라 행동을 선택하여 반환

    def replay(self, batch_size):
        if len(self.memory) < self.replay_start_size:
            # 메모리에 저장된 경험이 충분하지 않으면 리플레이를 수행하지 않습니다.
            return  

        # 미니배치를 랜덤으로 샘플링합니다.
        minibatch = random.sample(self.memory, batch_size)
        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            # 상태와 다음 상태를 패딩하고 마스킹합니다.
            state, mask = pad_state_action(state, [])
            next_state, next_mask = pad_state_action(next_state, [])
            state = torch.FloatTensor(state).unsqueeze(0)
            mask = torch.FloatTensor(mask).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            next_mask = torch.FloatTensor(next_mask).unsqueeze(0)
            
            target = reward  # 타겟 값 초기화
            if not done:
                # 다음 상태에서의 최적 행동에 대한 타겟 네트워크의 예측을 사용하여 타겟 값을 업데이트합니다.
                best_action = torch.argmax(self.model(next_state, next_mask)[0]).item()
                target += self.gamma * self.target_model(next_state, next_mask)[0][best_action].item()
                                        #target_model2였는데 target_model으로
            target_f = self.model(state, mask)  # 현재 상태에서의 예측값
            target_f[0][action] = target  # 타겟 값을 현재 상태의 예측값으로 업데이트
            self.optimizer.zero_grad()  # 옵티마이저의 그래디언트 초기화
            loss = self.criterion(target_f, self.model(state, mask))  # 손실 계산
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
    def __init__(self, max_state_size, action_size, gnn, update_target_frequency=32, replay_start_size=1000):
        # TaskAgent 클래스의 생성자. DQNAgent를 상속받으며, GNN 인스턴스를 추가로 인자로 받습니다.
        super(TaskAgent, self).__init__(max_state_size, action_size, update_target_frequency, replay_start_size)
        self.gnn = gnn  # GNN 인스턴스를 클래스 변수로 저장

    def act(self, state, mask=None, env=None):
        graph_data = create_graph(env)  # 환경에서 그래프 데이터를 생성
        gnn_output = self.gnn(graph_data)  # GNN을 통해 그래프 정보를 얻음
        gnn_output_flat = gnn_output.flatten().detach().numpy()  # GNN 출력을 플래튼하고 넘파이 배열로 변환

        # print(f"TaskAgent State shape: {state.shape}, TaskAgent GNN output shape: {gnn_output_flat.shape}")

        extended_state = np.concatenate((state, gnn_output_flat))  # 기존 상태 벡터와 GNN 출력을 결합
        padded_state, mask = pad_state_action(extended_state, [])  # 결합된 상태를 패딩 처리
        
        return super(TaskAgent, self).act(padded_state, mask)  # 결합된 상태를 기반으로 행동 선택


class MachineAgent(DQNAgent):
    def __init__(self, max_state_size, action_size, gnn, update_target_frequency=32, replay_start_size=1000):
        # MachineAgent 클래스의 생성자. DQNAgent를 상속받으며, GNN 인스턴스를 추가로 인자로 받습니다.
        super(MachineAgent, self).__init__(max_state_size, action_size, update_target_frequency, replay_start_size)
        self.gnn = gnn  # GNN 인스턴스를 클래스 변수로 저장

    def act(self, state, mask=None, env=None):
        graph_data = create_graph(env)  # 환경에서 그래프 데이터를 생성
        gnn_output = self.gnn(graph_data)  # GNN을 통해 그래프 정보를 얻음
        gnn_output_flat = gnn_output.flatten().detach().numpy()  # GNN 출력을 플래튼하고 넘파이 배열로 변환

        # print(f"MachineAgent State shape: {state.shape}, MachineAgent GNN output shape: {gnn_output_flat.shape}")
        
        extended_state = np.concatenate((state, gnn_output_flat))  # 기존 상태 벡터와 GNN 출력을 결합
        padded_state, mask = pad_state_action(extended_state, [])  # 결합된 상태를 패딩 처리
        
        return super(MachineAgent, self).act(padded_state, mask)  # 결합된 상태를 기반으로 행동 선택


def pad_state_action(state, action, max_state_size=1800, pad_value=-2):
    # 상태와 행동을 패딩하는 함수. 주어진 최대 상태 크기와 패딩 값을 사용합니다.
    if len(state) > max_state_size:
        # 상태 벡터가 최대 크기를 초과하면 잘라냅니다.
        state = state[:max_state_size]
    
    max_action_size = 30 * 10  # 최대 행동 크기 설정
    padded_state = np.full(max_state_size, pad_value)  # 패딩 값을 사용하여 최대 상태 크기의 배열 생성
    padded_state[:len(state)] = state  # 상태 벡터를 패딩된 배열에 복사
    mask = np.zeros(max_state_size)  # 마스크 배열 생성 (0으로 초기화)
    mask[:len(state)] = 1  # 상태 벡터의 길이까지 마스크를 1로 설정
    padded_action = np.full(max_action_size, pad_value)  # 패딩 값을 사용하여 최대 행동 크기의 배열 생성
    padded_action[:len(action)] = action  # 행동 벡터를 패딩된 배열에 복사
    return padded_state, mask  # 패딩된 상태와 마스크 반환

def train_individual_models(datasets, num_episodes_per_dataset):
    # 상태 크기와 행동 크기의 최대값을 설정합니다.
    max_state_size = 30 * 10 * 6
    action_size = 30 * 10

    # GNN의 입력, 숨김, 출력 차원을 설정합니다.
    gnn_input_dim = 2
    gnn_hidden_dim_1 = 64 # 기존 128
    gnn_hidden_dim_2 = 32 # 신규 추가
    gnn_hidden_dim_3 = 16 # 신규 추가
    gnn_output_dim = 8 # 기존 64

    # GNN 모델을 생성합니다.
    gnn = GNN(gnn_input_dim, gnn_hidden_dim_1,gnn_hidden_dim_2,gnn_hidden_dim_3, gnn_output_dim)

    # TaskAgent와 MachineAgent를 생성합니다. #update_target_frequency = 32,replay_start_size=40
    task_agent = TaskAgent(max_state_size, action_size, gnn, update_target_frequency=64, replay_start_size=100)
    machine_agent = MachineAgent(max_state_size, action_size, gnn, update_target_frequency=64, replay_start_size=100)

    # 각 데이터셋에 대해 학습을 수행합니다.
    for i, dataset in enumerate(datasets):
        print(f"Starting training for dataset {dataset} ({i+1}/{len(datasets)})")

        # 주어진 경로에서 데이터셋을 로드합니다.
        data = RLDataset_FJSSP(dataset)

        # 작업 처리 시간과 기계 순서를 준비합니다.
        process_times = [[(m, t) for m, t in ops] for job in data.op_data for ops in job]
        machine_sequence = [[[(m, t) for m, t in ops] for ops in job] for job in data.op_data]

        for job in range(len(machine_sequence)):
            print(f"Job {job}: {machine_sequence[job]}")

        # 환경 객체를 생성합니다.
        env = JobShopEnv_FJSSP(process_times, machine_sequence, solutions=None)

        # 상태와 행동의 크기를 계산합니다.
        state_size = env.n_jobs * env.n_machines * 3
        action_size = env.n_jobs * env.n_machines

        # 주어진 에피소드 수만큼 학습을 수행합니다.
        for episode in range(num_episodes_per_dataset):
            state = env.reset()
            state, mask = pad_state_action(np.concatenate((
                np.array([state[0]]), 
                state[1], 
                state[2], 
                state[3],  # 추가된 기계 평균 이용률
                state[4],  # 추가된 작업 큐 길이
                state[5]   # 추가된 작업 진행률
            )), [])
            max_op_counts = [len(ops) for ops in machine_sequence]
            job_op_counter = [0] * env.n_jobs

            done = False
            episode_reward = 0
            while not done:
                task_action = task_agent.act(state, mask, env)
                selected_job, selected_op = divmod(task_action, env.n_machines)

                if selected_job >= env.n_jobs or selected_op >= max_op_counts[selected_job] or job_op_counter[selected_job] != selected_op:
                    continue

                machine_action = machine_agent.act(state, mask, env)
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
                    next_state[3],  # 추가된 기계 평균 이용률
                    next_state[4],  # 추가된 작업 큐 길이
                    next_state[5]   # 추가된 작업 진행률
                )), [])

                task_agent.remember(state, selected_job * env.n_machines + selected_op, step_reward, next_state, done)
                machine_agent.remember(state, selected_op * env.n_jobs + selected_job, step_reward, next_state, done)

                state, mask = next_state, next_mask

                job_op_counter[selected_job] += 1
                if job_op_counter[selected_job] > max_op_counts[selected_job]:
                    job_op_counter[selected_job] = max_op_counts[selected_job]

                episode_reward += step_reward

                if done or all(count >= max_op_counts[j] for j, count in enumerate(job_op_counter)):
                    reward_task, reward_machine = env.calculate_episode_rewards()
                    print(f"Training: Episode {episode+1} processed. Reward Task: {reward_task}, Reward Machine: {reward_machine}, Step Reward: {episode_reward}")
                    break

            if len(task_agent.memory) >= task_agent.replay_start_size:
                for _ in range(10): # 기존 range 4, 32
                    task_agent.replay(64) # batch_size
                    machine_agent.replay(64)

            if task_agent.update_counter % task_agent.update_target_frequency == 0:
                task_agent.update_target_model()
                print("Updated target network for task agent during training.")
            
            if machine_agent.update_counter % machine_agent.update_target_frequency == 0:
                machine_agent.update_target_model()
                print("Updated target network for machine agent during training.")

            # 현재 에피소드가 완료되었음을 출력합니다.
            if episode % 1 == 0:
                print(f"Episode {episode}/{num_episodes_per_dataset} completed for dataset {dataset}")

        print(f"Completed training for dataset {dataset} ({i+1}/{len(datasets)})")

    return task_agent, machine_agent, max_state_size, action_size


def predict(task_agent, machine_agent, env, test_dataset, num_predictions=1, max_steps=100000000):
    # 예측 과정에서 유효한 솔루션 수를 추적하기 위한 변수 초기화
    valid_solution_count = 0
    # 모든 예측된 솔루션을 저장하기 위한 리스트 초기화
    all_predictions = []
    # 스텝 수를 추적하기 위한 변수 초기화
    step_count = 0

    # 최대 예측 수에 도달하거나 최대 스텝 수를 초과할 때까지 반복
    while valid_solution_count < num_predictions and step_count < max_steps:
        # 환경을 초기 상태로 리셋
        state = env.reset()
        # 최대 작업 단계 수를 복사하여 초기화
        max_op_counts = test_dataset.max_op_counts.copy()
        # 각 작업의 현재 작업 단계 수를 추적하기 위한 딕셔너리 초기화
        job_op_counter = {job: 0 for job in range(env.n_jobs)}

        # 초기 상태에 분석된 특성 값을 추가하여 확장
        state = np.concatenate((
            np.array([state[0]]),
            state[1],
            state[2],
            state[3],  # 추가된 기계 평균 이용률
            state[4],  # 추가된 작업 큐 길이
            state[5]  # 추가된 작업 진행률
            # state[6]   # 추가된 작업 남은 시간
        ))
        # 상태를 패딩하고 마스킹하여 최대 상태 크기에 맞춤
        state, mask = pad_state_action(state, [], max_state_size=1800)

        # 현재 솔루션을 저장할 리스트 초기화
        solution = []
        # 에피소드 완료 여부를 나타내는 변수 초기화
        done = False

        # 초기 최대 작업 단계 수를 출력
        print(f"Initial max_op_counts: {max_op_counts}")

        # 에피소드가 완료되지 않았고 최대 스텝 수를 초과하지 않을 때까지 반복
        while not done and step_count < max_steps:
            # 현재 스텝의 시작 시간을 기록
            start_time = time.time()
            
            # Step 1: Task agent가 처리할 작업-작업 단계를 결정
            task_action = task_agent.act(state, mask, env)
            # 작업-작업 단계 인덱스를 작업 번호와 작업 단계 번호로 분리
            job, op = divmod(task_action, env.n_machines)

            # 선택된 작업과 작업 단계가 유효한지 확인
            if job >= env.n_jobs or op >= len(env.machine_sequence[job]) or job_op_counter[job] != op:
                continue

            # Step 2: Machine agent가 선택된 작업-작업 단계를 처리할 기계를 결정
            machine_action = machine_agent.act(state, mask, env)
            # 기계 인덱스를 기계 번호로 변환
            machine = machine_action % env.n_machines

            # 선택된 작업, 작업 단계, 기계를 솔루션에 추가
            machine_options = env.machine_sequence[job][op]
            # 선택된 기계에 대한 처리 시간을 찾음
            machine_duration = next((t for m, t in machine_options if m == machine), None)
            if machine_duration is None:
                continue

            # 현재 작업 단계의 선택을 솔루션 리스트에 추가
            solution.append((job, op, machine, machine_duration))
            # 해당 작업의 작업 단계 수를 증가
            job_op_counter[job] += 1
            # 총 스텝 수를 증가
            step_count += 1

            # 만약 작업의 현재 작업 단계 수가 최대 작업 단계 수를 초과하면, 완료 표시
            if job_op_counter[job] > max_op_counts[job]:
                max_op_counts[job] = -1
                print(f"Job {job} completed. Updated max_op_counts: {max_op_counts}")

            # 100 스텝마다 현재 상태를 출력하고 가비지 컬렉션 수행
            if step_count % 100 == 0:
                print(f"Predicting: Step {step_count}, Current Solution: {solution[-5:]}")
                gc.collect()

            # 환경에서 선택된 작업, 작업 단계, 기계를 사용하여 스텝을 진행하고 새로운 상태를 얻음
            next_state, done, step_reward = env.step(job, op, machine)
            # 새로운 상태에 분석된 특성 값을 추가하여 확장
            state = np.concatenate((
                np.array([next_state[0]]),
                next_state[1],
                next_state[2],
                next_state[3], # 추가된 기계 평균 이용률
                next_state[4], # 추가된 작업 큐 길이
                next_state[5] # 추가된 작업 진행률
                # next_state[6] # 추가된 작업 남은 시간
            ))
                
            # 상태를 패딩하고 마스킹하여 최대 상태 크기에 맞춤
            state, mask = pad_state_action(state, [], max_state_size=1800)
            # 현재 스텝의 종료 시간을 기록
            end_time = time.time()
            # 현재 스텝 수와 작업 단계 카운터를 출력
            print(f"Step {step_count}, job_op_counter: {job_op_counter}")
            print(f"Step idle time reward: {step_reward}")

        # 에피소드가 완료되면 보상을 계산하고 출력
        if done:
            reward_task, reward_machine = env.calculate_episode_rewards()
            print(f"Prediction: Episode {valid_solution_count+1}/{num_predictions} processed. Reward Task: {reward_task}, Reward Machine: {reward_machine}")
            # 유효한 솔루션 수를 증가시키고 현재 솔루션을 모든 예측 리스트에 추가
            valid_solution_count += 1
            all_predictions.append(solution)
            # 가비지 컬렉션 수행
            gc.collect()

            # 환경을 초기 상태로 리셋하고 최대 작업 단계 수를 초기화
            env.reset()
            max_op_counts = test_dataset.max_op_counts.copy()
            # 각 작업의 현재 작업 단계 수를 초기화
            job_op_counter = {job: 0 for job in range(env.n_jobs)}

    # 유효한 솔루션 수가 원하는 예측 수에 도달하지 못한 경우 경고 출력
    if valid_solution_count != num_predictions:
        print(f"Warning: Only {valid_solution_count} valid solutions were collected.")

    # 모든 예측된 솔루션을 반환
    return all_predictions


# main 함수
def main():
    # 데이터셋 경로를 지정. 다양한 데이터셋을 사용하도록 수정
    datasets = [
        'fjsspdataset/HurinkRdata4.fjs',
        'fjsspdataset/HurinkRdata5.fjs',
        'fjsspdataset/HurinkRdata6.fjs',
    ]
    num_episodes_per_dataset = 20  # 각 데이터셋당 에피소드 수 설정

    # 개별 모델을 훈련시키고, task_agent와 machine_agent를 반환받음
    task_agent, machine_agent, max_state_size, action_size = train_individual_models(datasets, num_episodes_per_dataset)

    # 테스트 데이터셋 로드. 이 예제에서는 하나의 테스트 데이터셋을 사용
    test_dataset = RLDataset_FJSSP('fjsspdataset/HurinkRdata7.fjs')

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
       

