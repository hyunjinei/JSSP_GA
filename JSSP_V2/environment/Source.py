import simpy
import numpy as np
from environment.Part import Job


# region Source
class Source(object):
    def __init__(self, _env, _name, _model, _monitor, part_type, op_data, config,
                 IAT='exponential(1)'):
        self.env = _env
        self.name = _name  # 해당 Source의 이름
        self.model = _model
        self.monitor = _monitor
        self.part_type = part_type  # Source가 생산하는 Part의 type
        self.IAT = config.IAT  # Source가 생성하는 Part의 IAT(jobtype을 통한 Part 생성)
        # self.num_parts = config.num_parts  # Source가 생성하는 Part의 갯수
        self.num_parts = float('inf')
        self.op_data = op_data
        self.config = config

        self.rec = 0  # 생성된 Part의 갯수를 기록하는 변수
        self.generated_parts = simpy.Store(_env, capacity=10)  # 10 is an arbitrary number

        _env.process(self.generate())
        _env.process(self.routing())

    def generate(self):
        while True:
            while self.rec < self.num_parts:
                # 1. Generate a Part Object
                part = Job(self.env, self.part_type, self.rec, self.op_data)
                part.loc = self.name  # Update the part's current location

                # 2. Update the number of parts generates
                # so that the Source would stop after generating a certain amount of parts
                self.generated_parts.put(part)
                self.rec += 1

                # 3. Record through monitor class
                self.monitor.record(time=self.env.now, process=self.name, machine=None,
                                    part_name=part.name,
                                    event="Part" + str(self.part_type) + " Created")

                # 4. Print through Console (Optional)
                if self.config.print_console:
                    print('-' * 15 + part.name + " Created" + '-' * 15)
                # 5. Proceed on IAT timeout
                # ! Handling an IAT value given as a string variable
                # If self.IAT is the string 'exponential(1)',
                # then this line will be equivalent to IAT = np.random.exponential(1)
                if type(self.IAT) is str:
                    IAT = eval('np.random.' + self.IAT)
                else:
                    IAT = self.IAT
                yield self.env.timeout(np.round(IAT))

    def routing(self):
        while True:
            while self.rec < self.num_parts:
                # 1. Get a part from the list of generated parts
                part = yield self.generated_parts.get()
                part.step += 1  # this makes part.step to 0
                self.monitor.record(self.env.now, self.name, machine=None,
                                    part_name=part.name,
                                    event=str(part.name) + "_Routing Start")

                # 2. Check the next process
                # The machine is not assigned yet and is to be determined further, in the 'Process' class function
                next_process = self.model['Process' + str(part.op[part.step].process_type)]  # i.e. model['Process0']

                # 3. Put the part into the in_part queue of the next process
                # This 'yield' enables handling Process of limited queue,
                # by pending the 'put' call until the process is available for a new part
                if self.config.print_console:
                    print(part.name, "is going to be put in", next_process.name)


                yield next_process.in_part.put(part)
                part.loc = next_process.name
                next_process.input_event.succeed()  # Enables detection of incoming part
                next_process.input_event = simpy.Event(self.env)

                # 4. Record
                self.monitor.record(self.env.now, self.name, machine=None,
                                    part_name=part.name,
                                    event=part.name+"_Routing Finished")
                self.monitor.record(self.env.now, next_process.name, machine=None,
                                    part_name=part.name, event=part.name+" transferred from Source")


# endregion
