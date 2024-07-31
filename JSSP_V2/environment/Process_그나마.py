import simpy
from .Monitor import *

class Process(object):
    def __init__(self, _env, _name, _model, _monitor, _job_order, config):
        self.config = config
        self.env = _env
        self.name = _name
        self.model = _model
        self.monitor = _monitor
        self.job_order = _job_order
        self.parts_sent = 0

        self.in_part = simpy.Store(_env)
        self.out_part = simpy.Store(_env)

        _env.process(self.work())

    def work(self):
        machine_end_times = [0] * self.config.n_machine
        for job in self.job_order:
            part = yield self.in_part.get(lambda x: x.part_type == job)
            
            for step in range(self.config.n_machine):
                machine = self.model['M' + str(step)]
                operation = part.op[step]
                process_time = operation.process_time

                start_time = max(self.env.now, machine_end_times[step])
                yield self.env.timeout(start_time - self.env.now)
                yield machine.availability.put('using')

                try:
                    self.monitor.record(start_time, self.name, machine='M' + str(step),
                                        part_name=part.name, event="Started")
                except Exception as e:
                    print(f"Error logging 'Started' event: {e}")

                end_time = start_time + process_time
                yield self.env.timeout(process_time)

                try:
                    self.monitor.record(end_time, self.name, machine='M' + str(step),
                                        part_name=part.name, event="Finished")
                except Exception as e:
                    print(f"Error logging 'Finished' event: {e}")

                machine.util_time += process_time
                machine_end_times[step] = end_time
                yield machine.availability.get()

            yield self.out_part.put(part)
            self.parts_sent += 1

        while self.parts_sent < len(self.job_order):
            yield self.env.timeout(1)

        self.model['Sink'].put(part)