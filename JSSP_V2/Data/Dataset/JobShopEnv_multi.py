def calculate_makespan_reward(solution, dataset):
    job_start_times = {job: 0 for job in range(dataset.n_jobs)}
    machine_avail_times = {machine: 0 for machine in range(dataset.n_machines)}
    makespan = 0

    for job, op in solution:
        machine, duration = dataset.process_times[job][op]
        start_time = max(job_start_times[job], machine_avail_times[machine])
        end_time = start_time + duration
        job_start_times[job] = end_time
        machine_avail_times[machine] = end_time
        makespan = max(makespan, end_time)

    return makespan

def calculate_idle_time(agent_actions, env):
    machine_end_times = {m: 0 for m in range(env.n_machines)}
    job_start_times = {job: 0 for job in range(env.n_jobs)}
    machine_idle_times = {m: 0 for m in range(env.n_machines)}

    for job, op in agent_actions:
        machine = env.machine_sequence[job][op]
        processing_time = env.process_times[job][op][1]
        start_time = max(job_start_times[job], machine_end_times[machine])
        end_time = start_time + processing_time

        # 현재 작업이 시작될 때까지의 idle time 추가
        if start_time > machine_end_times[machine]:
            idle_time = start_time - machine_end_times[machine]
            machine_idle_times[machine] += idle_time
            # print(f"Job {job}, Op {op}, Machine {machine}: Idle Time Added {idle_time}")

        machine_end_times[machine] = end_time
        job_start_times[job] = end_time

    # 모든 머신의 idle time을 합산
    total_idle_time = sum(machine_idle_times.values())
    
    # print(f"Total Idle Times by Machine: {machine_idle_times}")
    return total_idle_time

def calculate_waiting_time(agent_actions, env):
    waiting_time = 0
    job_start_times = {job: 0 for job in range(env.n_jobs)}
    machine_end_times = {machine: 0 for machine in range(env.n_machines)}
    job_operations = {job: [] for job in range(env.n_jobs)}  # 각 작업의 연산 순서를 저장하는 딕셔너리 추가

    for job, op in agent_actions:
        machine = env.machine_sequence[job][op]
        processing_time = env.process_times[job][op][1]
        start_time = max(job_start_times[job], machine_end_times[machine])
        
        # 대기 시간 계산: 현재 작업의 시작 시간 - 작업의 이전 종료 시간
        job_waiting_time = start_time - job_start_times[job]
        waiting_time += job_waiting_time

        # 작업의 대기 시간 출력
        job_operations[job].append((op, job_waiting_time))
        
        end_time = start_time + processing_time
        machine_end_times[machine] = end_time
        job_start_times[job] = end_time
    
    # 작업 대기 시간을 순서대로 출력
    for job, ops in job_operations.items():
        ops.sort()  # 연산 순서대로 정렬
        # for op, wait_time in ops:
        #     print(f"Job {job}, Op {op}, Waiting Time Added {wait_time}")

    return waiting_time

class JobShopEnv:
    def __init__(self, process_times, machine_sequence, solutions=None, analysis_values=None):
        self.n_jobs = len(process_times)
        self.n_machines = len(process_times[0])
        self.process_times = process_times
        self.machine_sequence = machine_sequence
        self.solutions = solutions
        self.analysis_values = analysis_values
        self.optimal_solution = None
        self.previous_makespan = None  # 이전 에피소드의 makespan을 저장할 변수 추가
        self.previous_idle_time = None  # 이전 에피소드의 유휴 시간을 저장할 변수 추가
        self.previous_waiting_time = None  # 이전 에피소드의 대기 시간을 저장할 변수 추가
        if solutions is not None and len(solutions) > 0:
            self.solution_actions = set((divmod(a - 1, self.n_machines)) for solution in self.solutions for a in solution)
            self.optimal_solution = [divmod(a - 1, self.n_machines) for a in solutions[0]]
        self.reset()

    def reset(self):
        self.current_time = 0
        self.job_completion = [0] * self.n_jobs
        self.machine_available_time = [0] * self.n_machines
        self.machine_task_completion = [0] * self.n_machines
        self.agent_actions = []
        self.rewards = []  # 보상 저장
        self.state = (self.current_time, tuple(self.job_completion), tuple(self.machine_available_time))
        return self.state
    
    def step(self, job, op, use_solution_actions=False):
        if op >= len(self.process_times[job]):
            raise ValueError(f"Invalid operation index: job={job}, op={op}")

        machine = self.machine_sequence[job][op]
        processing_time = self.process_times[job][op][1]
        start_time = max(self.current_time, self.machine_available_time[machine])
        end_time = start_time + processing_time

        self.machine_available_time[machine] = end_time
        self.job_completion[job] += 1
        self.current_time = max(self.machine_available_time)

        self.agent_actions.append((job, op))

        done = all(c == self.n_machines for c in self.job_completion)

        self.state = (self.current_time, tuple(self.job_completion), tuple(self.machine_available_time))
        return self.state, done

    def calculate_episode_rewards(self, is_pretrain=False):
        total_reward_task = 0
        total_reward_machine = 0

        makespan = calculate_makespan_reward(self.agent_actions, self)
        idle_time = calculate_idle_time(self.agent_actions, self)
        waiting_time = calculate_waiting_time(self.agent_actions, self)
        
        # print(f'Makespan: {makespan}')
        # print(f'Idle Time: {idle_time}')
        # print(f'Waiting Time: {waiting_time}')

        reward_task = -makespan - waiting_time * 0.3  # 대기 시간을 반영하여 보상을 조정
        reward_machine = -makespan - idle_time

        # 이전 makespan과 비교하여 보상을 조정
        if self.previous_makespan is not None:
            makespan_diff = self.previous_makespan - makespan
            reward_task += makespan_diff * 2
            # print(f'Makespan difference: {makespan_diff}, Adjusted Reward Task: {reward_task}')

        # 이전 유휴 시간과 비교하여 보상을 조정
        if self.previous_idle_time is not None:
            idle_time_diff = self.previous_idle_time - idle_time
            reward_machine += idle_time_diff * 2
            # print(f'Idle Time difference: {idle_time_diff}, Adjusted Reward Machine: {reward_machine}')
        
        # 이전 대기 시간과 비교하여 보상을 조정
        if self.previous_waiting_time is not None:
            waiting_time_diff = self.previous_waiting_time - waiting_time
            reward_task += waiting_time_diff * 0.2
            # print(f'Waiting Time difference: {waiting_time_diff}, Adjusted Reward Task: {reward_task}')
        
        # 현재 makespan, 유휴 시간, 대기 시간을 이전 값으로 업데이트
        self.previous_makespan = makespan
        self.previous_idle_time = idle_time
        self.previous_waiting_time = waiting_time

        # print(f'보상 reward_task 1: {reward_task}')
        # print(f'보상 reward_machine 1: {reward_machine}')

        total_reward_task += reward_task
        total_reward_machine += reward_machine

        if self.optimal_solution is not None:
            for idx, (job, op) in enumerate(self.agent_actions):
                if idx < len(self.optimal_solution) and self.optimal_solution[idx] == (job, op):
                    total_reward_task += 2
                    # print(f"보상 부여: 현재 인덱스 {idx}, 최적 솔루션 {self.optimal_solution[idx]}, 에이전트 수행 {(job, op)}, 보상 reward_task 3: {total_reward_task}")
                # else:
                #     print(f"일치하지 않음: 현재 인덱스 {idx}, 최적 솔루션 {self.optimal_solution[idx] if idx < len(self.optimal_solution) else 'None'}, 에이전트 수행 {(job, op)}")

            if self.agent_actions == self.optimal_solution:
                total_reward_task += 10000
                # print("전체 순서와 작업이 일치하여 추가 보상 부여")
                
        if is_pretrain:
            solution_makespan = calculate_makespan_reward(self.optimal_solution, self)
            makespan_diff = solution_makespan - calculate_makespan_reward(self.agent_actions, self)
            total_reward_task -= makespan_diff * 2  # 패널티 부여
            # print(f"Pretraining makespan comparison. Solution Makespan: {solution_makespan}, Makespan Diff: {makespan_diff}")

        print(f'총 보상 reward_task: {total_reward_task}')
        print(f'총 보상 reward_machine: {total_reward_machine}')

        return total_reward_task, total_reward_machine
