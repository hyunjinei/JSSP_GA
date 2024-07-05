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

# 로그 파일 설정
logging.basicConfig(filename='training_progress.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class DQN(nn.Module):
    def __init__(self, max_state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(max_state_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, action_size)

    def forward(self, x):
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
        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            state, _ = pad_state_action(state, [])
            next_state, _ = pad_state_action(next_state, [])
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

def analyze_solutions(solutions, op_data):
    job_intervals = []
    operation_gaps = []
    job_waiting_times = []
    machine_transition_times = []
    job_completion_ratios = []

    for solution in solutions:
        job_start_times = {job: [] for job in range(len(op_data))}
        machine_end_times = {machine: 0 for machine in range(len(op_data[0]))}
        job_wait_times = {job: 0 for job in range(len(op_data))}
        
        for step in solution:
            job, op = divmod(step - 1, len(op_data[0]))
            machine, duration = op_data[job][op]
            start_time = max(machine_end_times[machine], job_start_times[job][-1] + duration if job_start_times[job] else 0)
            end_time = start_time + duration
            
            job_start_times[job].append(start_time)
            machine_end_times[machine] = end_time

            if len(job_start_times[job]) > 1:
                prev_end_time = job_start_times[job][-2] + op_data[job][op-1][1]
                job_wait_times[job] += start_time - prev_end_time

        job_intervals.append([max(times) - min(times) if times else 0 for times in job_start_times.values()])
        operation_gaps.append([times[i] - times[i-1] if i > 0 else 0 for times in job_start_times.values() for i in range(len(times))])
        job_waiting_times.append([wait_time for wait_time in job_wait_times.values()])
        machine_transition_times.append([end_time for end_time in machine_end_times.values()])
        job_completion_ratios.append([sum(times) / sum([sum(times) for times in job_start_times.values()]) for times in job_start_times.values()])

    return {
        "job_intervals": job_intervals,
        "operation_gaps": operation_gaps,
        "job_waiting_times": job_waiting_times,
        "machine_transition_times": machine_transition_times,
        "job_completion_ratios": job_completion_ratios
    }

def pad_state_action(state, action):
    max_state_size = 100 * 20 * 3  # 100 job, 20 machine, 3 상태
    max_action_size = 100 * 20     # 100 job, 20 machine
    padded_state = np.zeros(max_state_size)
    padded_state[:len(state)] = state
    padded_action = np.zeros(max_action_size)
    padded_action[:len(action)] = action
    return padded_state, padded_action

def pretrain_with_solutions(agent, solutions, env, repeat_count=100):
    total_solutions = len(solutions)
    for i in range(repeat_count):
        solution = solutions[random.randint(0, total_solutions - 1)]
        state = env.reset()
        state, _ = pad_state_action(np.concatenate((np.array([state[0]]), state[1], state[2])), [])
        for j, action in enumerate(solution):
            job, op = divmod(action - 1, env.n_machines)
            if job >= env.n_jobs or op >= len(env.process_times[job]):
                continue
            next_state, reward, done = env.step(job, op)
            next_state, _ = pad_state_action(np.concatenate((np.array([next_state[0]]), next_state[1], next_state[2])), [])
            agent.remember(state, job * env.n_machines + op, reward, next_state, done)
            state = next_state
            if done:
                break
        
        for _ in range(2):
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

def train_individual_models(datasets):
    models = []
    max_state_size = 100 * 20 * 3  # 최대 상태 크기
    max_action_size = 100 * 20     # 최대 액션 크기
    
    for i, dataset in enumerate(datasets):
        logging.info(f"Starting training for dataset {dataset} ({i+1}/{len(datasets)})")
        print(f"Starting training for dataset {dataset} ({i+1}/{len(datasets)})")
        data = RLDataset(dataset)
        solutions = load_solutions(f"{data.name}_solutions.txt", max_solutions=100000)
        env = JobShopEnv(data.op_data, [[m for m, _ in job] for job in data.op_data], solutions)

        state_size = env.n_jobs * env.n_machines * 3
        action_size = env.n_jobs * env.n_machines

        agent = DQNAgent(max_state_size, action_size)
        repeat_count = 1440 if len(solutions) < 100000 else len(solutions)
        pretrain_with_solutions(agent, solutions, env, repeat_count=repeat_count)
        
        for _ in range(100):
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

    while valid_solution_count < num_predictions and step_count < max_steps:
        state = env.reset()
        state, _ = pad_state_action(np.concatenate((np.array([state[0]]), state[1], state[2])), [])
        solution = []
        done = False
        job_op_counter = {job: 0 for job in range(env.n_jobs)}
        total_operations = env.n_jobs * env.n_machines

        while not done and step_count < max_steps:
            action = agent.act(state)
            job, op = divmod(action, env.n_machines)
            
            if job >= env.n_jobs or op >= env.n_machines or job_op_counter[job] != op:
                continue
            
            state, _, done = env.step(job, op, use_solution_actions=False)
            state, _ = pad_state_action(np.concatenate((np.array([state[0]]), state[1], state[2])), [])
            solution.append((job, op))
            job_op_counter[job] += 1
            step_count += 1
            
            if step_count % 1000 == 0:
                print(f"Predicting: Step {step_count}, Current Solution: {solution[-5:]}")
                gc.collect()
            
            done = all(op == env.n_machines for op in job_op_counter.values())

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
    test_dataset = RLDataset('la03.txt')
    combined_model = combine_models(models, max_state_size, max_action_size)
    logging.info("Training completed for individual models and combined model created.")
    print("Training completed for individual models and combined model created.")

    env = JobShopEnv(test_dataset.op_data, [[m for m, _ in job] for job in test_dataset.op_data])
    all_predictions = predict(combined_model, env, test_dataset, num_predictions=10, max_steps=100000000)
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

