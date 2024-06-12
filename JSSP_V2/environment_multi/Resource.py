import simpy

class Machine:
    def __init__(self, env, idx):
        self.env = env
        self.idx = idx
        self.availability = simpy.Resource(env, capacity=1)
        self.util_time = 0
        self.tasks = []

    def add_reference(self, operation, part_name):
        try:
            job_id_str, op_id_str = part_name.split('_')
            job_id = int(job_id_str[4:])  # 'Part' 이후의 숫자
            op_id = int(op_id_str)
            task = {"job_id": job_id, "op_id": op_id, "start_time": None, "end_time": None}
            self.tasks.append(task)
        except (IndexError, ValueError) as e:
            raise ValueError(f"Error parsing part_name '{part_name}': {e}")

    def calculate_idle_time_and_makespan(self):
        idle_time = 0
        makespan = 0
        real_time = 0

        if not self.tasks:
            return idle_time, makespan, real_time

        # 첫 작업 시작 전의 유휴 시간 계산
        idle_time += self.tasks[0]["start_time"]
        for i, task in enumerate(self.tasks):
            if task["end_time"] > makespan:
                makespan = task["end_time"]
            real_time += task["end_time"] - task["start_time"]
            if i < len(self.tasks) - 1:
                idle_time += self.tasks[i + 1]["start_time"] - task["end_time"]

        return idle_time, makespan, real_time





class Worker(object):
    def __init__(self, env, id):
        self.env = env
        self.id = id
        self.capacity = 1
        self.availability = simpy.Store(env, capacity=self.capacity)
        self.workingtime_log = []
        self.util_time = 0.0


class Jig(object):
    def __init__(self, env, id):
        self.env = env
        self.id = id
        self.capacity = 1
        self.availability = simpy.Store(env, capacity=self.capacity)
        self.workingtime_log = []
        self.util_time = 0.0
