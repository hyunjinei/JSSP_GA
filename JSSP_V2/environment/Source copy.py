import simpy
from environment.Part import Job

class Source(object):
    def __init__(self, _env, _name, _model, _monitor, job_order, op_data, config):
        self.env = _env
        self.name = _name
        self.model = _model
        self.monitor = _monitor
        self.job_order = job_order
        self.op_data = op_data
        self.config = config

        self.parts_generated = 0

        _env.process(self.generate())

    def generate(self):
        for job in self.job_order:
            part = Job(self.env, job, self.parts_generated, self.op_data)
            part.loc = self.name

            self.monitor.record(time=self.env.now, process=self.name, machine=None,
                                part_name=part.name,
                                event=f"Part{job} Created")

            if self.config.print_console:
                print('-' * 15 + part.name + " Created" + '-' * 15)

            next_process_index = part.op[0].process_type
            next_process = self.model['Process' + str(next_process_index)]
            yield next_process.in_part.put(part)

            self.monitor.record(self.env.now, self.name, machine=None,
                                part_name=part.name,
                                event=part.name + "_Routing Finished")
            self.monitor.record(self.env.now, next_process.name, machine=None,
                                part_name=part.name, event=part.name + " transferred from Source")

            self.parts_generated += 1
            yield self.env.timeout(1)  # 1초 간격으로 job 생성
