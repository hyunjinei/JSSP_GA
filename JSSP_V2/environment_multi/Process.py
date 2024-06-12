import simpy
from .Monitor import *

# Process.py
class Process(object):
    def __init__(self, _env, _name, _model, _monitor, _machine_order, config):
        self.config = config
        self.env = _env
        self.name = _name
        self.model = _model
        self.monitor = _monitor
        self.machine_order = _machine_order
        self.parts_sent = 0
        self.scheduled = 0

        self.in_part = simpy.FilterStore(_env)
        self.part_ready = simpy.FilterStore(_env)
        self.out_part = simpy.FilterStore(_env)
        self.input_event = simpy.Event(_env)
        self.ready_event = simpy.Event(_env)
        self.route_ready = simpy.Event(_env)

        _env.process(self.work())
        _env.process(self.routing())
        _env.process(self.dispatch())

    def work(self):
        while True:
            part = yield self.part_ready.get()
            operation = part.op[part.step]
            yield operation.requirements

            if isinstance(operation.machine, list):
                machine = operation.machine[0]
                process_time = operation.process_time[0]
            else:
                machine = self.model['M' + str(operation.machine)]
                process_time = operation.process_time

            with machine.availability.request() as req:
                yield req
                machine.add_reference(operation, part.name)

                # 로깅 추가: Started 이벤트
                try:
                    self.monitor.record(self.env.now, self.name, machine='M' + str(operation.machine),
                                        part_name=part.name, event="Started")
                except Exception as e:
                    print(f"Error logging 'Started' event: {e}")

                start_time = self.env.now
                yield self.env.timeout(process_time)
                end_time = self.env.now

                # 로깅 추가: Finished 이벤트
                try:
                    self.monitor.record(self.env.now, self.name, machine='M' + str(operation.machine),
                                        part_name=part.name, event="Finished")
                except Exception as e:
                    print(f"Error logging 'Finished' event: {e}")

                machine.util_time += process_time
                self.input_event.succeed()

                # Update machine task times
                for task in machine.tasks:
                    if task["job_id"] == int(part.name.split('_')[0][4:]) and task["op_id"] == int(part.name.split('_')[1]):
                        task["start_time"] = start_time
                        task["end_time"] = end_time

                yield self.out_part.put(part)


    def dispatch(self):
        while True:
            yield self.input_event
            self.input_event = simpy.Event(self.env)
            if self.config.dispatch_mode == 'FIFO':
                part_ready = yield self.in_part.get()
                yield self.part_ready.put(part_ready)
            elif self.config.dispatch_mode == 'Manual':
                num_scan = len(self.in_part.items)
                for i in range(num_scan):
                    if self.check_item():
                        part_ready = yield self.in_part.get(lambda x: x.part_type == self.machine_order[self.scheduled])
                        yield self.part_ready.put(part_ready)
                        self.scheduled += 1

    def check_item(self):
        for i, item in enumerate(self.in_part.items):
            if item.part_type == self.machine_order[self.scheduled]:
                return True
        return False

    def routing(self):
        while True:
            part = yield self.out_part.get()
            if part.step != (self.config.n_machine - 1):
                part.step += 1
                part.op[part.step].requirements.succeed()
                next_process = self.model['Process' + str(part.op[part.step].process_type)]
                yield next_process.in_part.put(part)
                next_process.input_event.succeed()
                next_process.input_event = simpy.Event(self.env)
                part.loc = next_process.name
            else:
                self.model['Sink'].put(part)
