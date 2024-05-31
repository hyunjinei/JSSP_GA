import copy
import random
import numpy as np

class PSO:
    def __init__(self, num_particles=30, num_iterations=100, w=0.7, c1=1.5, c2=1.5):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.w = w  # 관성 계수
        self.c1 = c1  # 개인 최적 위치로 이동하는 계수
        self.c2 = c2  # 전체 최적 위치로 이동하는 계수

    def optimize(self, individual, config):
        print("PSO 시작")

        # Initialize particles
        particles = [copy.deepcopy(individual) for _ in range(self.num_particles)]
        velocities = [np.random.uniform(-1, 1, len(individual.seq)) for _ in range(self.num_particles)]
        personal_best_positions = [copy.deepcopy(p.seq) for p in particles]
        personal_best_fitness = [p.calculate_fitness(config.target_makespan) for p in particles]
        
        global_best_particle = min(particles, key=lambda p: p.fitness)
        global_best_position = global_best_particle.seq[:]
        global_best_fitness = global_best_particle.fitness

        for iteration in range(self.num_iterations):
            for i in range(self.num_particles):
                r1 = random.random()
                r2 = random.random()
                
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (np.array(personal_best_positions[i]) - np.array(particles[i].seq)) +
                                 self.c2 * r2 * (np.array(global_best_position) - np.array(particles[i].seq)))
                velocities[i] = np.clip(velocities[i], -1, 1)
                
                particles[i].seq = np.array(particles[i].seq) + velocities[i]
                particles[i].seq = np.clip(particles[i].seq, 0, len(individual.seq) - 1)
                particles[i].seq = particles[i].seq.astype(int).tolist()

                # 시퀀스 유효성 검사 및 수정
                particles[i].seq = self.ensure_valid_sequence(particles[i].seq, len(individual.seq))

                fitness = particles[i].calculate_fitness(config.target_makespan)
                
                if fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = particles[i].seq[:]
                    personal_best_fitness[i] = fitness

                if fitness < global_best_fitness:
                    global_best_particle = copy.deepcopy(particles[i])
                    global_best_position = particles[i].seq[:]
                    global_best_fitness = fitness

            print(f"Iteration {iteration}: Global best fitness = {global_best_fitness}")

        print("PSO 종료")
        return global_best_particle

    def ensure_valid_sequence(self, seq, length):
        # 시퀀스에서 중복된 값 제거 및 누락된 값 추가
        valid_seq = list(set(seq))
        missing_values = [i for i in range(length) if i not in valid_seq]
        valid_seq.extend(missing_values)
        return valid_seq[:length]
