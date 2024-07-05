class MultiAgentGraph:
    def __init__(self, machines, jobs):
        self.machines = machines
        self.jobs = jobs
        self.graph = self.build_graph()

    def build_graph(self):
        graph = {}
        # Initialize machine nodes
        for machine in self.machines:
            graph[f'M_{machine}'] = {
                'type': 'machine',
                'static_edges': [],
                'dynamic_edges': {'executing': [], 'waiting': [], 'routing': []}
            }
        # Initialize job nodes
        for job in self.jobs:
            graph[f'J_{job}'] = {
                'type': 'job',
                'dynamic_edges': {'executing': [], 'waiting': [], 'routing': []}
            }
        return graph

    def update_graph(self, state, op, machine):
        # Clear dynamic edges
        for node in self.graph.values():
            if node['type'] == 'machine':
                node['dynamic_edges'] = {'executing': [], 'waiting': [], 'routing': []}
            elif node['type'] == 'job':
                node['dynamic_edges'] = {'executing': [], 'waiting': [], 'routing': []}

        # Update dynamic edges based on current state
        for job_id, job_info in state['jobs'].items():
            job_key = f'J_{job_id}'  # 이미 'J_'가 포함된 형식
            current_machine = job_info['current_machine']
            if current_machine is not None:
                machine_key = f'M_{current_machine}'
                self.graph[machine_key]['dynamic_edges']['executing'].append(job_key)
                self.graph[job_key]['dynamic_edges']['executing'].append(machine_key)
            for machine_id in job_info['waiting_machines']:
                machine_key = f'M_{machine_id}'
                self.graph[machine_key]['dynamic_edges']['waiting'].append(job_key)
                self.graph[job_key]['dynamic_edges']['waiting'].append(machine_key)
            for machine_id in job_info['routing_machines']:
                machine_key = f'M_{machine_id}'
                self.graph[machine_key]['dynamic_edges']['routing'].append(job_key)
                self.graph[job_key]['dynamic_edges']['routing'].append(machine_key)

    def get_state_features(self, agent_id):
        if self.graph[agent_id]['type'] == 'machine':
            features = self.get_machine_features(agent_id)
        elif self.graph[agent_id]['type'] == 'job':
            features = self.get_job_features(agent_id)
        
        # Convert the features dictionary to a list
        features_list = list(features.values())
        return features_list

    def get_machine_features(self, machine_id):
        machine_node = self.graph[machine_id]
        features = {
            'executing_jobs': len(machine_node['dynamic_edges']['executing']),
            'waiting_jobs': len(machine_node['dynamic_edges']['waiting']),
            'routing_jobs': len(machine_node['dynamic_edges']['routing'])
        }
        return features

    def get_job_features(self, job_id):
        job_node = self.graph[job_id]
        features = {
            'executing_machines': len(job_node['dynamic_edges']['executing']),
            'waiting_machines': len(job_node['dynamic_edges']['waiting']),
            'routing_machines': len(job_node['dynamic_edges']['routing'])
        }
        return features
