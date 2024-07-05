class JobShopEnv:
    def __init__(self, process_times, machine_sequence, solutions=None, analysis_values=None):
        self.n_jobs = len(process_times)
        self.n_machines = len(process_times[0])
        self.process_times = process_times
        self.machine_sequence = machine_sequence
        self.solutions = solutions
        self.analysis_values = analysis_values
        if solutions is not None:
            self.solution_actions = set((divmod(a - 1, self.n_machines)) for solution in self.solutions for a in solution)
        self.reset()

    def reset(self):
        self.current_time = 0
        self.job_completion = [0] * self.n_jobs
        self.machine_available_time = [0] * self.n_machines
        self.state = (self.current_time, tuple(self.job_completion), tuple(self.machine_available_time))
        return self.state

    def step(self, job, op, use_solution_actions=False):
        if op >= len(self.process_times[job]):
            raise ValueError(f"Invalid operation index: job={job}, op={op}")

        machine = self.machine_sequence[job][op]
        processing_time = self.process_times[job][op][1]
        start_time = max(self.current_time, self.machine_available_time[machine])
        end_time = start_time + processing_time

        previous_current_time = self.current_time

        self.machine_available_time[machine] = end_time
        self.job_completion[job] += 1
        self.current_time = max(self.machine_available_time)

        reward = 0
        
        # Makespan 감소 보상
        reward += (previous_current_time - self.current_time) * 100

        # 작업 완료 시 추가 보상
        done = all(c == self.n_machines for c in self.job_completion)
        if done:
            reward += 50 # 모든 작업 완료 시 추가 보상

        # 솔루션 일치 보상
        if use_solution_actions and hasattr(self, 'solution_actions'):
            if (job, op) in self.solution_actions:
                reward += 3000000  # 예시: 일치하는 경우 더 높은 보상 부여

        # 작업 대기 시간 감소 보상
        current_waiting_time = sum(self.machine_available_time) - previous_current_time * len(self.machine_available_time)
        next_waiting_time = current_waiting_time - processing_time
        reward += (current_waiting_time - next_waiting_time) * 10  # 대기 시간 감소 보상

        # 기계 유휴 시간 감소 보상
        for m in range(self.n_machines):
            if self.machine_available_time[m] < self.current_time:
                reward -= (self.current_time - self.machine_available_time[m]) * 10  # 유휴 시간 감소 보상

        # 기계 사용률 보상
        machine_usage = [self.machine_available_time[m] / (self.current_time + 1) for m in range(self.n_machines)]
        reward += sum(1 - usage for usage in machine_usage) * 50  # 기계 사용률 보상

        # 작업 완료 시점 보상
        if self.job_completion[job] == self.n_machines:
            reward += 50000  # 작업 완료 시 추가 보상

        # 작업 순서 보상
        if self.job_completion[job] == op + 1:
            reward += 10  # 작업 순서 보상
        # done = all(c == self.n_machines for c in self.job_completion)

        self.state = (self.current_time, tuple(self.job_completion), tuple(self.machine_available_time))
        return self.state, reward, done

    def get_valid_actions(self):
        valid_actions = []
        for job in range(self.n_jobs):
            if self.job_completion[job] < self.n_machines:
                valid_actions.append((job, self.job_completion[job]))
        return valid_actions
