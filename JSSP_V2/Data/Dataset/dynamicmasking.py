import os
import pandas as pd
from RLDataset import RLDataset
from JobShopEnv import JobShopEnv
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
# 로그 파일 설정
logging.basicConfig(filename='training_progress.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class DQN(nn.Module):
    def __init__(self, max_state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(max_state_size, 128)
        self.fc2 = nn.Linear(128, 128)               
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask  # 마스크 적용
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, max_state_size, action_size):
        self.state_size = max_state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(max_state_size, action_size)
        self.target_model = DQN(max_state_size, action_size)
        self.target_model2 = DQN(max_state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model2.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, mask=None):
        state = torch.FloatTensor(state).unsqueeze(0)
        if mask is not None:
            mask = torch.FloatTensor(mask).unsqueeze(0)
        act_values = self.model(state, mask).detach().numpy()[0]
        
        # Soft ε-greedy 정책 적용
        max_action = np.argmax(act_values)
        probabilities = np.ones(self.action_size) * (self.epsilon / self.action_size)
        probabilities[max_action] += (1.0 - self.epsilon)
        
        return np.random.choice(self.action_size, p=probabilities)



    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            state, mask = pad_state_action(state, [])
            next_state, next_mask = pad_state_action(next_state, [])
            state = torch.FloatTensor(state).unsqueeze(0)
            mask = torch.FloatTensor(mask).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            next_mask = torch.FloatTensor(next_mask).unsqueeze(0)
            
            # Double DQN
            target = reward
            if not done:
                best_action = torch.argmax(self.model(next_state, next_mask)[0]).item()
                target += self.gamma * self.target_model2(next_state, next_mask)[0][best_action].item()
            
            target_f = self.model(state, mask)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state, mask))
            loss.backward()
            self.optimizer.step()
            
            if idx % 10 == 0:  # 로그 빈도를 줄임
                logging.info(f"Replay: Step {idx+1}/{batch_size} processed. Loss: {loss.item()}")
                print(f"Replay: Step {idx+1}/{batch_size} processed. Loss: {loss.item()}")
                gc.collect()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def load_solutions(filename, max_solutions=100000):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    if len(lines) <= max_solutions:
        solutions = [list(map(int, line.strip().split())) for line in lines]
    else:
        lines = random.sample(lines, max_solutions)
        solutions = [list(map(int, line.strip().split())) for line in lines]
    
    return solutions
    
def pad_state_action(state, action, pad_value=-1):
    max_state_size = 100 * 20 * 3
    max_action_size = 100 * 20
    padded_state = np.full(max_state_size, pad_value)
    padded_state[:len(state)] = state
    mask = np.zeros(max_state_size)
    mask[:len(state)] = 1
    padded_action = np.full(max_action_size, pad_value)
    padded_action[:len(action)] = action
    return padded_state, mask

def pretrain_with_solutions(agent, solutions, env, repeat_count=100):
    total_solutions = len(solutions)
    for i in range(repeat_count):
        solution = solutions[random.randint(0, total_solutions - 1)]
        state = env.reset()
        state, mask = pad_state_action(np.concatenate((np.array([state[0]]), state[1], state[2])), [])
        for j, action in enumerate(solution):
            job, op = divmod(action - 1, env.n_machines)
            if job >= env.n_jobs or op >= len(env.process_times[job]):
                continue
            next_state, reward, done = env.step(job, op, use_solution_actions=True) #solution 일치 보상
            next_state, next_mask = pad_state_action(np.concatenate((np.array([next_state[0]]), next_state[1], next_state[2])), [])
            agent.remember(state, job * env.n_machines + op, reward, next_state, done)
            state, mask = next_state, next_mask
            if done:
                break
        
        for _ in range(1):
            agent.replay(32)
        
        if i % 100 == 0:
            logging.info(f"Pretraining: Solution {i+1}/{repeat_count} processed.")
            print(f"Pretraining: Solution {i+1}/{repeat_count} processed.")
            gc.collect()
    agent.update_target_model()

def combine_models(models, max_state_size, max_action_size):
    combined_model = models[0]
    combined_state_dict = combined_model.model.state_dict()

    for key in combined_state_dict.keys():
        combined_weight = combined_state_dict[key].clone()
        for model in models[1:]:
            model_weight = model.model.state_dict()[key]
            if combined_weight.size() == model_weight.size():
                combined_weight += model_weight
            else:
                logging.warning(f"Skipping combining weights for layer {key} due to size mismatch: {combined_weight.size()} vs {model_weight.size()}")
        combined_state_dict[key] = combined_weight / len(models)

    combined_model.model.load_state_dict(combined_state_dict)
    combined_model.state_size = max_state_size
    combined_model.action_size = max_action_size
    logging.info("Models combined into a single model.")
    print("Models combined into a single model.")
    return combined_model

def analyze_data(op_data):
    first_op_machines = [job[0][0] for job in op_data]
    last_op_machines = [job[-1][0] for job in op_data]
    
    machine_op_distribution = {m: [0] * len(op_data[0]) for m in range(len(op_data[0]))}
    op_durations = {op: 0 for op in range(len(op_data[0]))}
    max_duration = 0

    for job in op_data:
        for op_index, (machine, duration) in enumerate(job):
            print(f"Machine: {machine}, Duration: {duration}")
            machine_op_distribution[machine][op_index] += 1
            op_durations[op_index] += duration
            if duration > max_duration:
                max_duration = duration

    op_durations_total = sum(op_durations.values())
    
    machine_op_ratios = {m: [op / len(op_data) for op in ops] for m, ops in machine_op_distribution.items()}
    op_time_ratios_total = {op: duration / op_durations_total for op, duration in op_durations.items()}
    op_time_ratios_max = {op: duration / max_duration for op, duration in op_durations.items()}

    cumulative_time = 0
    
    for op in range(len(op_data[0])):
        cumulative_time += op_durations[op]

    first_op_machine_ratio = len(set(first_op_machines)) / len(op_data[0])
    last_op_machine_ratio = len(set(last_op_machines)) / len(op_data[0])

    return {
        "first_op_machine_ratio": first_op_machine_ratio,
        "last_op_machine_ratio": last_op_machine_ratio,
        "machine_op_ratios": machine_op_ratios,
        "op_time_ratios_total": op_time_ratios_total,
        "op_time_ratios_max": op_time_ratios_max,
    }
    
def train_individual_models(datasets):
    models = []
    max_state_size = 100 * 20 * 3  # 최대 상태 크기
    max_action_size = 100 * 20     # 최대 액션 크기
    
    for i, dataset in enumerate(datasets):
        logging.info(f"Starting training for dataset {dataset} ({i+1}/{len(datasets)})")
        print(f"Starting training for dataset {dataset} ({i+1}/{len(datasets)})")
        
        data = RLDataset(dataset)
        solutions = load_solutions(f"{data.name}_solutions.txt", max_solutions=100000)
        
        # 데이터 분석 추가
        op_data_analysis = analyze_data(data.op_data)
        print(f"Data Analysis Results: {op_data_analysis}")

        # 분석 결과를 리스트로 변환
        analysis_values = []
        for key, value in op_data_analysis.items():
            if isinstance(value, dict):
                analysis_values.extend(value.values())
            else:
                analysis_values.append(value)

        env = JobShopEnv(data.op_data, [[m for m, _ in job] for job in data.op_data], solutions, analysis_values)

        state_size = env.n_jobs * env.n_machines * 3
        action_size = env.n_jobs * env.n_machines

        agent = DQNAgent(max_state_size, action_size)
        repeat_count = 5 if len(solutions) < 100000 else len(solutions) # 숫자수정하셈
        pretrain_with_solutions(agent, solutions, env, repeat_count=repeat_count)
        
        for _ in range(10):
            agent.replay(64)
        
        models.append(agent)
        logging.info(f"Completed training for dataset {dataset} ({i+1}/{len(datasets)})")
        print(f"Completed training for dataset {dataset} ({i+1}/{len(datasets)})")
        gc.collect()  # 메모리 캐시 정리
    
    return models, max_state_size, max_action_size

def predict(agent, env, test_dataset, num_predictions=2000, max_steps=100000000):
    valid_solution_count = 0
    all_predictions = []
    step_count = 0

    # 데이터 분석 추가
    op_data_analysis = analyze_data(test_dataset.op_data)
    print(f"Data Analysis Results (test data): {op_data_analysis}")

    # 분석 결과를 일차원 배열로 변환
    analysis_values = []
    for key, value in op_data_analysis.items():
        if isinstance(value, dict):
            for v in value.values():
                if isinstance(v, list):
                    analysis_values.extend(v)
                else:
                    analysis_values.append(v)
        else:
            analysis_values.append(value)
    
    analysis_values = np.array(analysis_values)

    while valid_solution_count < num_predictions and step_count < max_steps:
        state = env.reset()
        state, mask = pad_state_action(np.concatenate((np.array([state[0]]), state[1], state[2], analysis_values)), [])
        solution = []
        done = False
        job_op_counter = {job: 0 for job in range(env.n_jobs)}
        total_operations = env.n_jobs * env.n_machines

        while not done and step_count < max_steps:
            start_time = time.time()  # 시작 시간 기록
            action = agent.act(state, mask)
            job, op = divmod(action, env.n_machines)
            
            if job >= env.n_jobs or op >= env.n_machines or job_op_counter[job] != op:
                continue
            
            state, _, done = env.step(job, op, use_solution_actions=False)
            state, mask = pad_state_action(np.concatenate((np.array([state[0]]), state[1], state[2], analysis_values)), [])
            solution.append((job, op))
            job_op_counter[job] += 1
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"Predicting: Step {step_count}, Current Solution: {solution[-5:]}")
                gc.collect()
            
            done = all(op == env.n_machines for op in job_op_counter.values())
            end_time = time.time()  # 끝 시간 기록
            # print(f"Step {step_count} took {end_time - start_time} seconds")  # 소요 시간 출력

        if len(solution) == total_operations:
            valid_solution_count += 1
            if valid_solution_count % 1 == 0:
                print(f"Valid Solution {valid_solution_count}/{num_predictions} collected.")
            all_predictions.append(solution)
            gc.collect()

    if valid_solution_count != num_predictions:
        print(f"Warning: Only {valid_solution_count} valid solutions were collected.")
    
    return all_predictions


def main():
    datasets = ['la03.txt']
    models, max_state_size, max_action_size = train_individual_models(datasets)
    test_dataset = RLDataset('ta21.txt')
    
    # 데이터 분석 추가
    op_data_analysis = analyze_data(test_dataset.op_data)
    print(f"Data Analysis Results: {op_data_analysis}")

    # 분석 결과를 리스트로 변환
    analysis_values = []
    for key, value in op_data_analysis.items():
        if isinstance(value, dict):
            analysis_values.extend(value.values())
        else:
            analysis_values.append(value)

    combined_model = combine_models(models, max_state_size, max_action_size)
    logging.info("Training completed for individual models and combined model created.")
    print("Training completed for individual models and combined model created.")

    # env 객체 생성 로그 추가
    logging.info("Creating environment object.")
    print("Creating environment object.")
    env = JobShopEnv(test_dataset.op_data, [[m for m, _ in job] for job in test_dataset.op_data], solutions=None, analysis_values=analysis_values)

    # predict 함수 실행 로그 추가
    logging.info("Starting prediction process.")
    print("Starting prediction process.")
    all_predictions = predict(combined_model, env, test_dataset, num_predictions=5, max_steps=100000000)
 
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

    ga_initial_population = [job * test_dataset.n_machine + op for job, op in best_solution]
    logging.info(f"GA Initial Population: {ga_initial_population}")
    print("GA Initial Population:", ga_initial_population)

    draw_gantt_chart(best_solution, test_dataset)


def calculate_makespan(solution, dataset):
    job_start_times = {job: 0 for job in range(dataset.n_job)}
    machine_avail_times = {machine: 0 for machine in range(dataset.n_machine)}
    makespan = 0

    for job, op in solution:
        machine, duration = dataset.op_data[job][op]
        start_time = max(job_start_times[job], machine_avail_times[machine])
        end_time = start_time + duration
        job_start_times[job] = end_time
        machine_avail_times[machine] = end_time
        makespan = max(makespan, end_time)

    return makespan

def draw_gantt_chart(predictions, dataset):
    job_start_times = {job: 0 for job in range(dataset.n_job)}
    machine_avail_times = {machine: 0 for machine in range(dataset.n_machine)}
    
    gantt_chart = []

    for job, op in predictions:
        machine, duration = dataset.op_data[job][op]
        start_time = max(job_start_times[job], machine_avail_times[machine])
        end_time = start_time + duration
        
        gantt_chart.append((machine, start_time, end_time))
        
        job_start_times[job] = end_time
        machine_avail_times[machine] = end_time

    fig, ax = plt.subplots()
    colors = plt.cm.tab20.colors

    for idx, (machine, start_time, end_time) in enumerate(gantt_chart):
        ax.barh(machine, end_time - start_time, left=start_time, color=colors[idx % len(colors)])

    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_yticks(range(dataset.n_machine))
    ax.set_yticklabels([f'Machine {i+1}' for i in range(dataset.n_machine)])
    plt.show()

if __name__ == "__main__":
    main()
