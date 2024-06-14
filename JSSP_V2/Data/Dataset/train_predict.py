from RLDataset import RLDataset, load_solutions
from JobShopEnv import JobShopEnv
from QLearningAgent import QLearningAgent

def pretrain_with_solutions(agent, solutions):
    for solution in solutions:
        for i in range(len(solution) - 1):
            current_action = (solution[i] // agent.n_machines, solution[i] % agent.n_machines)
            next_action = (solution[i + 1] // agent.n_machines, solution[i + 1] % agent.n_machines)
            state = (0, [0] * agent.n_jobs, [0] * agent.n_machines)
            next_state = (0, [1 if j == current_action[0] else 0 for j in range(agent.n_jobs)], [1 if m == current_action[1] else 0 for m in range(agent.n_machines)])
            reward = -1
            agent.update_q_value(state, current_action, reward, next_state, [next_action])

def train_q_learning(env, agent, episodes):
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.get_action(state, valid_actions)
            next_state, reward, done = env.step(*action)
            agent.update_q_value(state, action, reward, next_state, valid_actions)
            state = next_state

def main():
    dataset = RLDataset('la02.txt')
    process_times = dataset.get_process_times()
    machine_sequence = dataset.get_machine_sequence()
    solutions = load_solutions('la02_solutions.txt')

    env = JobShopEnv(process_times, machine_sequence)
    agent = QLearningAgent(len(process_times), len(process_times[0]))

    # 기존 솔루션 데이터로 프리트레이닝
    pretrain_with_solutions(agent, solutions)

    # 강화학습으로 미세 조정
    train_q_learning(env, agent, episodes=1000)

    # 예측
    state = env.reset()
    done = False
    solution = []
    while not done:
        valid_actions = env.get_valid_actions()
        action = agent.get_action(state, valid_actions)
        solution.append(action)
        state, _, done = env.step(*action)
    
    print(f"Predicted Solution: {solution}")

if __name__ == "__main__":
    main()
