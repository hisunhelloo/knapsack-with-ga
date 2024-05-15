import random
import time
import numpy as np

def population2(n, value, weights, capacity, popsize):
    # 새로운 pop 생성
    pop = []
    for i in range(popsize):
        chromosome = np.zeros(shape=n, dtype=int)
    
        while sum(chromosome * weights) <= capacity:
            temp_chromosome = chromosome.copy()
            k = random.choices(range(n))
            chromosome[k] = 1
        pop.append(temp_chromosome) 
    #print(pop)
    # print(len(pop))

    return np.array(pop)

def population(n, value, weights, capacity, popsize):

    # effect 계산, 룰렛휠 적용
    effect = value / weights
    total_effect = sum(effect)
    effect /= total_effect
    print(popsize)
    
    # 새로운 pop 생성
    pop = []
    for i in range(popsize):
        chromosome = np.zeros(shape=n, dtype=int)
    
        while sum(chromosome * weights) <= capacity:
            temp_chromosome = chromosome.copy()
            k = random.choices(range(n), weights=effect)
            chromosome[k] = 1
        pop.append(temp_chromosome) 
    #print(pop)
    # print(len(pop))

    return np.array(pop)
    

def cal_fitness(pop, new_value):
    fitness_score = [sum(pop[i]*new_value) for i in range(len(pop))]
    return np.array(fitness_score)


def crossover(n, popsize, pop, crossover_rate):
    childs = np.zeros((0, n))
    if random.random() < crossover_rate:
        parent1 = random.choice(pop)
        parent2 = random.choice(pop)
        split_index = random.randint(1, n-1)
        child1 = np.concatenate((parent1[:split_index], parent2[split_index:]),axis=0)
        child2 = np.concatenate((parent2[:split_index], parent1[split_index:]),axis=0)
        childs = np.array([child1, child2])

    return childs

def mutate(offspring, mutation_rate):
    if random.random() < mutation_rate:
        n = random.randint(0, len(offspring)-1)
        offspring[n] = 1 - offspring[n]
    return offspring


def offspring(n, value, weights, capacity, popsize, pop, crossover_rate, mutation_rate):
    offspring = crossover(n, popsize, pop, crossover_rate)
    if offspring is not None:
        for i in range(popsize // 2):
            childs = crossover(n, popsize, pop, crossover_rate)
            if childs is not None and len(childs) > 0:
                offspring = np.append(offspring, childs, axis=0)
            else:
                parent = random.choice(pop)
                offspring = np.append(offspring, [parent], axis=0)

    for i in range(len(offspring)):
        offspring[i] = mutate(offspring[i], mutation_rate)
    
    # print(offspring)
    # print(len(offspring))
    # print(type(pop[0][0]), type(offspring[0][0]))
    
    for i in range(len(offspring)):
        temp_fit = sum(offspring[i] * weights)
        if temp_fit < capacity:
            while (temp_fit < capacity):
                j = random.randint(0, n-1)
                if offspring[i][j] == 0:
                    offspring[i][j] = 1 - offspring[i][j]
                    temp_fit += offspring[i][j]*weights[j]
            offspring[i][j] = 1 - offspring[i][j]
        else:
            while (temp_fit > capacity):
                j = random.randint(0, n-1)
                if offspring[i][j] == 1:
                    temp_fit -= offspring[i][j]*weights[j]
                    offspring[i][j] = 1 - offspring[i][j]
            offspring[i][j] = 1 - offspring[i][j]

    return offspring

def new_pop(population, offspring, value, popsize, weights):
    temp = np.concatenate((population, offspring), axis=0)

    fitness = cal_fitness(temp, value)
    
    new_population = []
    new_fitness = []
    for i in range(popsize):
        #토너먼트
        chosen_list = [random.choice(fitness) for _ in range(5)]
        i = list(fitness).index(max(chosen_list))

        new_population.append(temp[i])
        new_fitness.append(fitness[i])
    
    return np.array(new_population), np.array(new_fitness)


if __name__ == "__main__":

    # 주어진 값
    capacity = 200
    n = 30
    weights = [89, 45, 83, 78, 4, 65, 37, 36, 52, 41, 6, 77, 79, 75, 62, 34, 31, 87, 39, 5, 29, 20, 100, 93, 83, 32, 67, 24, 94, 93]
    value = [84, 16, 37, 4, 53, 62, 81, 59, 21, 54, 99, 16, 19, 44, 40, 39, 65, 86, 91, 14, 82, 21, 46, 83, 3, 15, 3, 75, 5, 66]
    # opt = [0 0 0 0 1 0 1 1 0 1 1 0 0 0 0 1 1 0 1 1 1 0 0 0 0 0 0 1 0 0];

    weights = np.array(weights)
    value = np.array(value)

    popsize = 300
    crossover_rate = 0.8
    mutation_rate = 0.2
    
    start_time = time.time()
    
    # 시작
    ## 첫 번째 population

    pop = population(n, value, weights, capacity, popsize)
    # pop = population2(n, value, weights, capacity, popsize)
    # print(pop)
    fitness = cal_fitness(pop, value)
   

    # 세대 반복
    generation = 0
    last_solution = 0
    ll = 0
    while True:
        generation += 1
        child = offspring(n, value, weights, capacity, popsize, pop, crossover_rate, mutation_rate)
        pop, fitness = new_pop(pop, child, value, popsize, weights)

        index = list(fitness).index(max(fitness))
        best_solution = pop[index]

        print(f"{generation} generation's best solution: {fitness[index]}")

        # 종료 조건
        # 1
        if generation > 1:
            if ll < 100:
                if (last_best == fitness[index]).all():
                    ll += 1
                else: 
                    ll = 0
            else: 
                break
        last_best = fitness[index]

        #2
        if time.time() - start_time > 1000000:
            break
        
    print(f"optimal solution found({generation} generation): {fitness[index]}")
