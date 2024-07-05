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

# 로그 파일 설정
logging.basicConfig(filename='training_progress.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_model(next_state)[0]).item()
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def load_solutions(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    solutions = [list(map(int, line.strip().split())) for line in lines]
    return solutions

def analyze_data(op_data):
    first_op_machines = [job[0][0] for job in op_data]
    last_op_machines = [job[-1][0] for job in op_data]
    
    machine_op_distribution = {m: [0] * len(op_data[0]) for m in range(len(op_data[0]))}
    op_durations = {op: 0 for op in range(len(op_data[0]))}
    max_duration = 0

    for job in op_data:
        for op_index, (machine, duration) in enumerate(job):
            machine_op_distribution[machine][op_index] += 1
            op_durations[op_index] += duration
            if duration > max_duration:
                max_duration = duration

    op_durations_total = sum(op_durations.values())
    
    machine_op_ratios = {m: [op / len(op_data) for op in ops] for m, ops in machine_op_distribution.items()}
    op_time_ratios_total = {op: duration / op_durations_total for op, duration in op_durations.items()}
    op_time_ratios_max = {op: duration / max_duration for op, duration in op_durations.items()}

    cumulative_time_ratios = {}
    # cumulative_job_ratios = {}    
    cumulative_time = 0
    cumulative_jobs = 0
    
    for op in range(len(op_data[0])):
        cumulative_time += op_durations[op]
        cumulative_jobs += 1
        cumulative_time_ratios[op] = cumulative_time / op_durations_total
        # cumulative_job_ratios[op] = cumulative_jobs / len(op_data[0])    

    first_op_machine_ratio = len(set(first_op_machines)) / len(op_data[0])
    last_op_machine_ratio = len(set(last_op_machines)) / len(op_data[0])

    return {
        "first_op_machine_ratio": first_op_machine_ratio,
        "last_op_machine_ratio": last_op_machine_ratio,
        "machine_op_ratios": machine_op_ratios,
        "op_time_ratios_total": op_time_ratios_total,
        "op_time_ratios_max": op_time_ratios_max,
        "cumulative_time_ratios": cumulative_time_ratios
        # "cumulative_job_ratios": cumulative_job_ratios        
    }
def analyze_solutions(solutions, op_data):
    job_intervals = []
    operation_gaps = []
    job_waiting_times = []
    machine_transition_times = []
    job_completion_ratios = []
    # job_priorities = []
    # job_continuities = []

    for solution in solutions:
        job_start_times = {job: [] for job in range(len(op_data))}
        machine_end_times = {machine: 0 for machine in range(len(op_data[0]))}
        job_wait_times = {job: 0 for job in range(len(op_data))}
        job_continuity_count = {job: 0 for job in range(len(op_data))}
        
        for step in solution:
            job, op = divmod(step - 1, len(op_data[0]))
            machine, duration = op_data[job][op]
            start_time = max(machine_end_times[machine], sum(job_start_times[job]))
            end_time = start_time + duration
            
            job_start_times[job].append(start_time)
            machine_end_times[machine] = end_time

            if len(job_start_times[job]) > 1:
                prev_end_time = job_start_times[job][-2] + op_data[job][op-1][1]
                job_wait_times[job] += start_time - prev_end_time
                job_continuity_count[job] += 1 if start_time == prev_end_time else 0

        job_intervals.append([max(times) - min(times) for times in job_start_times.values()])
        operation_gaps.append([times[i] - times[i-1] if i > 0 else 0 for times in job_start_times.values() for i in range(len(times))])
        job_waiting_times.append([wait_time for wait_time in job_wait_times.values()])
        machine_transition_times.append([end_time for end_time in machine_end_times.values()])
        job_completion_ratios.append([sum(times) / sum([sum(times) for times in job_start_times.values()]) for times in job_start_times.values()])
        # job_priorities.append([1.0 / len(times) for times in job_start_times.values()])
        # job_continuities.append([count / len(times) if len(times) > 0 else 0 for count, times in zip(job_continuity_count.values(), job_start_times.values())])

    return {
        "job_intervals": job_intervals,
        "operation_gaps": operation_gaps,
        "job_waiting_times": job_waiting_times,
        "machine_transition_times": machine_transition_times,
        "job_completion_ratios": job_completion_ratios,
        # "job_priorities": job_priorities,
        # "job_continuities": job_continuities
    }

def pretrain_with_solutions(agent, solutions, env):
    for i, solution in enumerate(solutions):
        state = env.reset()
        state = np.concatenate((np.array([state[0]]), state[1], state[2]))
        for j, action in enumerate(solution):
            job, op = divmod(action - 1, env.n_machines)
            if job >= env.n_jobs or op >= len(env.process_times[job]):
                continue
            next_state, reward, done = env.step(job, op)
            next_state = np.concatenate((np.array([next_state[0]]), next_state[1], next_state[2]))
            agent.remember(state, job * env.n_machines + op, reward, next_state, done)
            state = next_state
            if done:
                break
        if i % 100 == 0:
            logging.info(f"Pretraining: Solution {i+1}/{len(solutions)} processed.")
            print(f"Pretraining: Solution {i+1}/{len(solutions)} processed.")
    agent.update_target_model()

def train_individual_models(datasets):
    models = []
    for i, dataset in enumerate(datasets):
        logging.info(f"Starting training for dataset {dataset} ({i+1}/{len(datasets)})")
        print(f"Starting training for dataset {dataset} ({i+1}/{len(datasets)})")
        data = RLDataset(dataset)
        solutions = load_solutions(f"{data.name}_solutions.txt")
        env = JobShopEnv(data.op_data, [[m for m, _ in job] for job in data.op_data], solutions)
        
        # 데이터 분석
        analysis = analyze_data(data.op_data)
        print("Data Analysis:", analysis)
        
        # 솔루션 분석 추가
        solution_analysis = analyze_solutions(solutions, data.op_data)
        print("Solution Analysis:", solution_analysis)
        
        state_size = env.n_jobs * env.n_machines * 3
        action_size = env.n_jobs * env.n_machines
        agent = DQNAgent(state_size, action_size)
        pretrain_with_solutions(agent, solutions, env)
        models.append(agent)
        logging.info(f"Completed training for dataset {dataset} ({i+1}/{len(datasets)})")
        print(f"Completed training for dataset {dataset} ({i+1}/{len(datasets)})")
    return models

def combine_models(models, target_action_size):
    combined_model = models[0]  # 첫 번째 모델을 사용하여 초기화
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
    combined_model.action_size = target_action_size  # action size를 target_action_size로 조정
    logging.info("Models combined into a single model.")
    print("Models combined into a single model.")
    return combined_model

def predict(agent, env, test_dataset, max_steps=10000):
    state = env.reset()
    state = np.concatenate((np.array([state[0]]), state[1], state[2]))
    solution = []
    done = False
    step_count = 0
    job_op_counter = {job: 0 for job in range(env.n_jobs)}
    total_operations = env.n_jobs * env.n_machines

    # 테스트 데이터 분석
    analysis = analyze_data(test_dataset.op_data)
    print("Test Data Analysis:", analysis)
    
    while not done and step_count < max_steps:
        action = agent.act(state)
        job, op = divmod(action, env.n_machines)
        
        # JSSP 제약조건을 준수하여 job과 op가 유효한지 확인
        if job >= env.n_jobs or op >= env.n_machines or job_op_counter[job] != op:
            continue
        
        state, _, done = env.step(job, op, use_solution_actions=False)
        state = np.concatenate((np.array([state[0]]), state[1], state[2]))
        solution.append((job, op))
        job_op_counter[job] += 1  # 해당 job의 작업 단계 증가
        step_count += 1
        
        if step_count % 100 == 0:  # 100 스텝마다 진행 상황 출력
            print(f"Predicting: Step {step_count}, Current Solution: {solution[-5:]}")
        
        # 모든 작업이 완료되었는지 확인
        done = all(op == env.n_machines for op in job_op_counter.values())
    
    # 모든 작업을 다 사용했는지 확인하고, 누락된 작업이 있으면 예외 처리
    if len(solution) != total_operations:
        raise ValueError("Not all operations are used in the solution. Please check the prediction logic.")

    return solution

def main():
    datasets = ['abz5.txt', 'abz6.txt', 'dmu32.txt', 'dmu33.txt','la02.txt', 'dmu35.txt', 'ft06.txt', 'ft10.txt', 'ft20.txt','la01.txt', 'la02.txt', 'la03.txt', 'la04.txt', 'la05.txt', 'la06.txt', 'la07.txt', 'la08.txt', 'la09.txt', 'la10.txt', 'la11.txt', 'la12.txt','la13.txt', 'la14.txt', 'la15.txt', 'la16.txt', 'la17.txt', 'la18.txt', 'la19.txt', 'la20.txt', 'la23.txt', 'la26.txt', 'la28.txt', 'la30.txt','la31.txt', 'la32.txt', 'la33.txt', 'la34.txt', 'la35.txt', 'orb01.txt', 'orb02.txt', 'orb03.txt', 'orb04.txt', 'orb05.txt', 'orb06.txt', 'orb07.txt','orb08.txt', 'orb09.txt', 'orb10.txt', 'swv16.txt', 'swv17.txt', 'swv18.txt', 'swv19.txt', 'swv20.txt', 'ta01.txt']
    models = train_individual_models(datasets)
    test_dataset = RLDataset('la01.txt')
    target_action_size = test_dataset.n_job * test_dataset.n_machine
    combined_model = combine_models(models, target_action_size)
    logging.info("Training completed for individual models and combined model created.")
    print("Training completed for individual models and combined model created.")

    # 예측 수행
    env = JobShopEnv(test_dataset.op_data, [[m for m, _ in job] for job in test_dataset.op_data])
    predictions = predict(combined_model, env, test_dataset)
    logging.info(f"Predicted Solution: {predictions}")
    print("Predicted Solution:", predictions)

    # Operation 수만큼 솔루션을 나눔
    num_operations = sum(len(job) for job in test_dataset.op_data)
    divided_solutions = [predictions[i:i + num_operations] for i in range(0, len(predictions), num_operations)]

    # 가장 좋은 솔루션 선택
    best_solution = None
    best_makespan = float('inf')
    for solution in divided_solutions:
        makespan = calculate_makespan(solution, test_dataset)
        if makespan < best_makespan:
            best_makespan = makespan
            best_solution = solution

    logging.info(f"Best Solution: {best_solution}")
    print("Best Solution:", best_solution)

    # GA 초기 인구 생성
    ga_initial_population = [job * test_dataset.n_machine + op for job, op in best_solution]  # 수정된 부분
    logging.info(f"GA Initial Population: {ga_initial_population}")
    print("GA Initial Population:", ga_initial_population)

    # 간트 차트 그리기
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

    # 간트 차트 그리기
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
