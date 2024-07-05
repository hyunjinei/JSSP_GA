# QLearningAgent.py

class QLearningAgent:
    def __init__(self, n_jobs, n_machines, solutions, env):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.q_table = {}
        self.solutions = solutions
        self.env = env

    def get_q_value(self, state, action):
        state_tuple = tuple(state)  # 상태를 튜플로 변환하여 해시 가능하게 만듦
        action_tuple = tuple(action)
        return self.q_table.get((state_tuple, action_tuple), 0.0)

    def update_q_value(self, state, action, reward, next_state, valid_actions):
        state_tuple = tuple(state)
        action_tuple = tuple(action)
        next_state_tuple = tuple(next_state)
        best_next_action = max((self.get_q_value(next_state_tuple, a) for a in valid_actions), default=0.0)
        current_q_value = self.get_q_value(state_tuple, action_tuple)
        new_q_value = current_q_value + 0.1 * (reward + 0.99 * best_next_action - current_q_value)
        self.q_table[(state_tuple, action_tuple)] = new_q_value

    def get_valid_actions(self, state):
        return self.env.get_valid_actions()

    def initialize_state(self):
        return self.env.reset()

    def take_action(self, state, action):
        next_state, reward, done = self.env.step(*action)

        # 보상 계산: 작업과 공정이 솔루션 데이터와 일치하면 더 높은 보상 부여
        solution_actions = [(divmod(a - 1, self.n_machines)) for solution in self.solutions for a in solution]
        if action in solution_actions:
            reward += 1000  # 예시: 일치하는 경우 더 높은 보상 부여
        # else:
        #     reward -= 1  # 예시: 일치하지 않는 경우 보상은 -1 (최소화 문제)

        # Makespan 계산 및 보상 반영
        if next_state[0] < state[0]:  # makespan이 줄어든 경우 추가 보상
            reward += 10
        elif next_state[0] == state[0]: # makespan이 똑같으면 패널티
            reward -= -1
        else:
            reward -= 100 # makespan이 더 안좋으면 패널티

        return next_state, reward, done

    def update_q_table(self, other_q_table):
        for key, value in other_q_table.items():
            if key in self.q_table:
                self.q_table[key] = (self.q_table[key] + value) / 2
            else:
                self.q_table[key] = value

    def ensure_valid_sequence(self, seq):
        num_jobs = self.n_jobs
        num_machines = self.n_machines
        job_counts = {job: 0 for job in range(num_jobs)}
        valid_seq = []

        for operation in seq:
            job = operation // num_machines
            if job_counts[job] < num_machines:
                valid_seq.append(job * num_machines + job_counts[job])
                job_counts[job] += 1

        for job in range(num_jobs):
            while job_counts[job] < num_machines:
                valid_seq.append(job * num_machines + job_counts[job])
                job_counts[job] += 1

        return valid_seq
