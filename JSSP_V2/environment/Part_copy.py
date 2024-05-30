

# region Operation
class Operation(object):
    """
    This class does not act as a process.
    Instead, this is just a member variable of a job that contains process info.
    This class is only used when generating a job sequence.
    The Process class is to be generated further.
    """

    def __init__(self, env, id, part_name, process_type, machine, process_time, requirements=None):

        self.id = id
        self.process_type = process_type
        self.process_time = process_time
        self.part_name = part_name
        self.name = part_name + '_Op' + str(id)
        self.job_ready = False  # Add job_ready attribute
        self.machine_ready = False  # Add machine_ready attribute

        # In the simplest Job Shop problem, process type is often coincide with the machine type itself.
        self.machine = machine

        # Handle both int and tuple id cases
        if isinstance(id, tuple):
            self.precedence = id[1]
        else:
            self.precedence = id

        if env is not None and requirements is None:
            self.requirements = env.event()
            if (isinstance(id, tuple) and id[1] == 0) or (id == 0):
                self.requirements.succeed()
        else:
            self.requirements = [env.event() for i in range(5)] if env is not None else None

        self.op_prior = None
        self.op_following = None

    def set_start_time(self, start_time):
        self.start_time = start_time
        self.end_time = self.start_time + self.process_time

    def run(self):
        yield self.env.timeout(self.process_time)

# endregion

# region Job
class Job(object):
    """
    A job is to be repeatedly generated in a source.
    (Job1_1, Job1_2, Job1_3, ..., Job1_100,
    Job2_1, Job2_2, Job2_3, ..., Job2_100,
    ...                         Job10_100)

    Member Variable : part_type, id
    """

    def __init__(self, env, part_type, id, op_data):
        self.part_type = part_type
        self.id = id
        self.name = 'Part' + str(part_type) + '_' + str(id)
        self.step = -1
        self.loc = None  # current location
        self.op = [Operation(env,
                             id=j, part_name=self.name,
                             process_type=op_data[part_type][j][0],
                             machine=op_data[part_type][j][0],
                             process_time=op_data[part_type][j][1],
                             requirements=None) for j in range(len(op_data[part_type]))]
