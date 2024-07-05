# Ptr_Net.py
import os
import random
import torch
import logging
import torch.nn as nn
import torch.optim as optim
from collections import deque
from RLDataset import RLDataset
from JobShopEnv import JobShopEnv
from actor import PtrNet1
from critic import PtrNet2

# 로그 파일 설정
logging.basicConfig(filename='training_progress.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class PtrNetAgent:
    def __init__(self, params, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = PtrNet1(params)
        self.critic = PtrNet2(params) if params["use_critic"] else None
        self.optimizer = optim.Adam(self.model.parameters(), lr=params["learning_rate"])
        if self.critic:
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params["learning_rate"])
            self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=params["memory_size"])
        self.params = params

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, device):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            action, _, _ = self.model(state, device)
        return action[0].cpu().numpy()

    def learn(self, device):
        if len(self.memory) < self.params["batch_size"]:
            return

        batch = random.sample(self.memory, self.params["batch_size"])
        states, actions, rewards, next_states, dones = zip(*batch)

        # 각 상태를 개별 텐서로 변환하여 리스트로 저장
        states = [state.unsqueeze(0).to(device) for state in states]
        next_states = [next_state.unsqueeze(0).to(device) for next_state in next_states]
        rewards = torch.FloatTensor(rewards).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)

        self.model.train()
        log_probs = []
        for state, action in zip(states, actions):
            logging.info(f"State shape before model: {state.size()}")
            _, log_prob, _ = self.model(state, device, action)
            log_probs.append(log_prob)

        log_probs = torch.cat(log_probs).squeeze()
        loss = -(log_probs * rewards).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logging.info(f"Actor loss: {loss.item()}")

        if self.critic:
            self.critic.train()
            pred_rewards = []
            for state in states:
                pred_reward = self.critic(state, device).squeeze()
                pred_rewards.append(pred_reward.unsqueeze(0))  # 0차원 텐서를 1차원으로 변환

            pred_rewards = torch.cat(pred_rewards)
            critic_loss = self.criterion(pred_rewards, rewards)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            logging.info(f"Critic loss: {critic_loss.item()}")

    def pretrain(self, solutions, env, device, repeat_count=100):
        total_solutions = len(solutions)
        for i in range(repeat_count):
            solution = solutions[random.randint(0, total_solutions - 1)]
            state = env.reset()
            total_reward = 0
            for job, op in solution:
                next_state, reward, done = env.step(job, op)
                reward += self.params["solution_match_reward"] if (job, op) in env.solution_actions else 0
                self.remember(state, (job, op), reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break
            logging.info(f"Total reward for pretraining solution {i+1}/{repeat_count}: {total_reward}")
        self.learn(device)

def load_solutions(filename, max_solutions=100000):
    with open(filename, 'r') as file:
        lines = file.readlines()

    if len(lines) <= max_solutions:
        solutions = [list(map(int, line.strip().split())) for line in lines]
    else:
        lines = random.sample(lines, max_solutions)
        solutions = [list(map(int, line.strip().split())) for line in lines]

    return solutions

def load_datasets(file_list):
    datasets = {}
    for file in file_list:
        dataset = RLDataset(file)
        solution_file = file.replace('.txt', '_solutions.txt')
        solutions = load_solutions(solution_file, max_solutions=100000)
        datasets[file] = (dataset, solutions)
    return datasets

def pretrain_with_solutions(agent, solutions, env, device, repeat_count=100):
    total_solutions = len(solutions)
    for i in range(repeat_count):
        solution = solutions[random.randint(0, total_solutions - 1)]
        state = env.reset()
        total_reward = 0
        for step in solution:
            job, op = divmod(step - 1, env.n_machines)
            next_state, reward, done = env.step(job, op)
            reward += agent.params["solution_match_reward"] if (job, op) in env.solution_actions else 0

            # 디버깅 출력 추가
            logging.info(f"State: {state}, Next State: {next_state}")

            state_vector = [state[0]] + list(state[1]) + list(state[2])
            next_state_vector = [next_state[0]] + list(next_state[1]) + list(next_state[2])

            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(device)
            next_state_tensor = torch.FloatTensor(next_state_vector).unsqueeze(0).to(device)

            agent.remember(state_tensor, (job, op), reward, next_state_tensor, done)
            state = next_state
            total_reward += reward
            if done:
                break
        logging.info(f"Total reward for pretraining solution {i+1}/{repeat_count}: {total_reward}")
        agent.learn(device)

def train_individual_models(datasets, params, device):
    models = []
    for filename, (dataset, solutions) in datasets.items():
        logging.info(f"Pretraining with dataset {filename}")
        env = JobShopEnv(dataset.op_data, [[m for m, _ in job] for job in dataset.op_data], solutions)

        # 각 데이터셋의 입력 크기를 기반으로 input_dim 설정
        example_state = env.reset()
        logging.info(f"Example state: {example_state}")
        input_dim = len(example_state[1]) + len(example_state[2]) + 1
        params["input_dim"] = input_dim
        logging.info(f"Calculated input_dim: {input_dim}")

        state_size = env.n_jobs * env.n_machines * 3
        action_size = env.n_jobs * env.n_machines
        params["num_of_process"] = env.n_machines  # Ensure the number of processes is set correctly
        params["state_size"] = state_size  # Ensure the state size is set correctly
        params["action_size"] = action_size  # Ensure the action size is set correctly
        params["n_process"] = env.n_machines  # Ensure n_process is set correctly

        agent = PtrNetAgent(params, state_size, action_size)
        agent.model.to(device)
        if agent.critic:
            agent.critic.to(device)
        pretrain_with_solutions(agent, solutions, env, device, repeat_count=params["repeat_count"])
        models.append(agent)
        torch.save(agent.model.state_dict(), f"ptrnet_model_{filename}.pth")
        if agent.critic:
            torch.save(agent.critic.state_dict(), f"ptrnet_critic_{filename}.pth")
    return models


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models

    def forward(self, state, device):
        predictions = []
        for model in self.models:
            model.model.eval()
            with torch.no_grad():
                action, _, _ = model.model(state, device)
                predictions.append(action)
        predictions = torch.stack(predictions)
        ensemble_action = torch.mode(predictions, dim=0).values
        return ensemble_action


def find_max_state_size(datasets):
    max_size = 0
    for dataset, _ in datasets:
        for job in dataset.op_data:
            for operation in job:
                state_size = len(operation)  # 각 작업의 길이를 추정
                if state_size > max_size:
                    max_size = state_size
    return max_size

def evaluate_model(models, benchmark_dataset, env, device, max_state_size, num_predictions=100):
    ensemble_predict = EnsembleModel(models)
    env.reset()
    predictions = []
    for _ in range(num_predictions):
        state = env.reset()
        done = False
        prediction = []
        while not done:
            # 패딩을 max_state_size에 맞게 설정
            padded_state = pad_state(state, max_state_size)
            state_tensor = torch.FloatTensor(padded_state).unsqueeze(0).to(device)
            action = ensemble_predict(state_tensor, device)
            action = action.cpu().numpy()
            next_state, reward, done = env.step(*action)
            prediction.append(action)
            state = next_state
        predictions.append(prediction)
    return predictions


def pad_state(state, max_size, pad_value=-1):
    # state가 튜플로 이루어져 있는지 확인하고, 각 요소를 리스트로 변환하여 평탄화합니다.
    flattened_state = [element for sublist in state for element in sublist] if isinstance(state[0], (list, tuple)) else list(state)
    # 패딩을 추가합니다.
    padded_state = flattened_state + [pad_value] * (max_size - len(flattened_state))
    return padded_state

def main():
    train_file_list = ['abz5.txt', 'abz6.txt', 'dmu32.txt']
    datasets = load_datasets(train_file_list)

    params = {
        "learning_rate": 1e-4,
        "num_episodes": 10000,
        "log_interval": 10,
        "save_interval": 100,
        "use_critic": True,
        "n_embedding": 128,
        "n_hidden": 128,
        "init_min": -0.08,
        "init_max": 0.08,
        "decode_type": "sampling",
        "n_glimpse": 1,
        "T": 1.0,
        "C": 10,
        "batch_size": 64,
        "memory_size": 200000,
        "solution_match_reward": 1000,
        "repeat_count": 1,
        "use_logit_clipping": True,
        "n_process": None
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = train_individual_models(datasets, params, device)
    
    # 앙상블 모델 생성
    combined_model = EnsembleModel(models)
    torch.save(combined_model, "ptrnet_combined_model.pth")

    benchmark_file = 'ta21.txt'
    benchmark_dataset = RLDataset(benchmark_file)
    env = JobShopEnv(benchmark_dataset.op_data, [[m for m, _ in job] for job in benchmark_dataset.op_data])
    
    # 최대 state 크기를 찾습니다.
    max_state_size = find_max_state_size(list(datasets.values()) + [(benchmark_dataset, None)])
    
    predictions = evaluate_model(models, benchmark_dataset, env, device, max_state_size)

    logging.info(f"Predictions for benchmark dataset {benchmark_file}: {predictions}")

if __name__ == "__main__":
    main()
