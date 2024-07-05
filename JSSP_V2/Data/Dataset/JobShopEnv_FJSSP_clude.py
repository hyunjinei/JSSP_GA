import numpy as np

class JobShopEnv_FJSSP:
    def __init__(self, op_data):
        self.n_jobs = len(op_data)
        self.n_machines = max(max(machine for machine, _ in job_ops) for job in op_data for job_ops in job) + 1
        self.machine_sequence = op_data
        self.process_times = [[[time for machine, time in job_ops] for job_ops in job] for job in op_data]
        self.reset()

    def reset(self):
        self.current_time = 0
        self.job_completion = [0] * self.n_jobs
        self.machine_available_time = [0] * self.n_machines
        self.job_op_status = [[True] + [False] * (len(job_ops) - 1) for job_ops in self.machine_sequence]
        self.machine_status = [False] * self.n_machines
        self.agent_actions = []
        return self.get_state()

    def get_state(self):
        return {
            'current_time': self.current_time,
            'job_completion': self.job_completion,
            'machine_available_time': self.machine_available_time,
            'machine_utilization': self.calculate_machine_utilization(),
            'job_queue_length': self.calculate_job_queue_length(),
            'job_progress': self.calculate_job_progress(),
            'remaining_job_time': self.calculate_remaining_job_time(),
            'job_op_status': self.job_op_status,
            'machine_status': self.machine_status
        }

    def calculate_machine_utilization(self):
        total_time = max(self.current_time, 1)  # 0으로 나누는 것을 방지
        return [sum(self.process_times[j][o] for j, o, m in self.agent_actions if m == i) / total_time for i in range(self.n_machines)]

    def calculate_job_queue_length(self):
        return [sum(1 for j in range(self.n_jobs) if self.job_op_status[j][self.job_completion[j]] and m in [m for m, _ in self.machine_sequence[j][self.job_completion[j]]]) for m in range(self.n_machines)]

    def calculate_job_progress(self):
        return [completion / len(self.machine_sequence[job]) for job, completion in enumerate(self.job_completion)]

    def calculate_remaining_job_time(self):
        return [sum(min(t for _, t in ops[self.job_completion[job]:]) for ops in self.machine_sequence[job]) for job in range(self.n_jobs)]

    def step(self, job, op, machine):
        if not self.job_op_status[job][op] or machine not in [m for m, _ in self.machine_sequence[job][op]]:
            return self.get_state(), -100, False

        processing_time = next(t for m, t in self.machine_sequence[job][op] if m == machine)
        start_time = max(self.current_time, self.machine_available_time[machine])
        end_time = start_time + processing_time

        self.current_time = max(self.current_time, end_time)
        self.machine_available_time[machine] = end_time
        self.job_completion[job] += 1
        self.job_op_status[job][op] = False
        if op + 1 < len(self.job_op_status[job]):
            self.job_op_status[job][op + 1] = True

        self.agent_actions.append((job, op, machine))

        done = all(completion == len(ops) for completion, ops in zip(self.job_completion, self.machine_sequence))
        reward = -end_time if done else 0

        self.update_machine_status()
        return self.get_state(), reward, done

    def update_machine_status(self):
        current_max_time = max(self.machine_available_time)
        threshold = current_max_time - 0.2 * (current_max_time - min(self.machine_available_time))
        self.machine_status = [time <= threshold for time in self.machine_available_time]

    def get_valid_actions(self):
        valid_actions = []
        for job in range(self.n_jobs):
            for op in range(len(self.job_op_status[job])):
                if self.job_op_status[job][op]:
                    for machine, _ in self.machine_sequence[job][op]:
                        valid_actions.append((job, op, machine))
        return valid_actions

    def create_graph(self):
        edges = []
        for job in range(self.n_jobs):
            for op in range(len(self.machine_sequence[job])):
                for machine, _ in self.machine_sequence[job][op]:
                    edges.append((job, self.n_jobs + machine))
                    edges.append((self.n_jobs + machine, job))
        return np.array(edges).T

    def calculate_makespan(self):
        return max(self.machine_available_time)

    def calculate_idle_time(self):
        return sum(self.machine_available_time) - self.n_machines * self.current_time