import random
import time
import matplotlib.pyplot as plt
import numpy as np

def population2(n, value, weights, capacity, popsize):
    # 새로운 pop 생성
    pop = []
    for i in range(popsize):
        chromosome = np.zeros(shape=n, dtype=int)
    
        while sum(chromosome * weights) <= capacity:
            k = random.choices(range(n))
            chromosome[k] = 1
        chromosome[k] = 0
        pop.append(chromosome) 

    return np.array(pop)

def population(n, value, weights, capacity, popsize):

    # effect 계산, 룰렛휠 적용
    effect = value / weights
    total_effect = sum(effect)
    effect /= total_effect
    
    # 새로운 pop 생성
    pop = []
    for i in range(popsize):
        chromosome = np.zeros(shape=n, dtype=int)
    
        while sum(chromosome * weights) <= capacity:
            temp_chromosome = chromosome.copy()
            k = random.choices(range(n), weights=effect)
            chromosome[k] = 1
        
        pop.append(temp_chromosome) 

    return np.array(pop)
    

def cal_fitness(pop, new_value):
    fitness_score = [sum(pop[i]*(new_value)) for i in range(len(pop))]
    return np.array(fitness_score)


def crossover(n, popsize, pop, crossover_rate):
    childs = np.empty((0, n), dtype=int)
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
            # else:
            #     parent = random.choice(pop)
            #     offspring = np.append(offspring, [parent], axis=0)
    for i in range(len(offspring)):
        offspring[i] = mutate(offspring[i], mutation_rate)
    
    for i in range(len(offspring)):
        temp_fit = sum(offspring[i] * weights)
        if temp_fit < capacity:
            while (temp_fit < capacity):
                j = random.randint(0, n-1)
                if offspring[i][j] == 0:
                    offspring[i][j] = 1 #- offspring[i][j]
                    temp_fit += weights[j]
            offspring[i][j] = 1 - offspring[i][j]
        else:
            while (temp_fit > capacity):
                j = random.randint(0, n-1)
                if offspring[i][j] == 1:
                    temp_fit -= weights[j]
                    offspring[i][j] = 0 #- offspring[i][j]
    
    return offspring

def new_pop(population, offspring, value, popsize, weights):
    temp = np.concatenate((population, offspring), axis=0)
    fit_factor = cal_fitness(temp, value)
    fitness = fit_factor / sum(fit_factor)
    
    new_population = []
    new_fitness = []
    for i in range(popsize):
        k = random.choices(range(len(temp)), weights = fitness)
        new_population.append(temp[k][0])
        new_fitness.append(fit_factor[k])
    
    return np.array(new_population), np.array(new_fitness)


if __name__ == "__main__":

    # 주어진 값
    capacity = 500
    n = 30
    weights = [23, 78, 1, 32, 84, 30, 78, 43, 65, 78, 72, 35, 38, 68, 14, 13, 77, 43, 56, 23, 78, 70, 52, 17, 16, 43, 32, 28, 34, 84]
    value = [50, 46, 21, 82, 2, 85, 36, 69, 2, 13, 55, 85, 27, 16, 63, 17, 71, 69, 37, 38, 90, 82, 1, 35, 48, 12, 52, 37, 2, 14]

    weights = np.array(weights)
    value = np.array(value)

    popsize = 100
    crossover_rate = 0.8
    mutation_rate = 0.1
    
    start_time = time.time()
    
    # 시작
    ## 첫 번째 population

    pop = population(n, value, weights, capacity, popsize)
    fitness = cal_fitness(pop, value)
   

    # 세대 반복
    generation = 0
    most_best = 0
    last_solution = 0
    ll = 0
    fit_list = []
    while True:
        generation += 1
        child = offspring(n, value, weights, capacity, popsize, pop, crossover_rate, mutation_rate)

        pop, fitness = new_pop(pop, child, value, popsize, weights)

        index = list(fitness).index(max(fitness))
        best_solution = fitness[index]
        fit_list.append(best_solution)
        print(f"{generation} generation's best solution:, {best_solution}")

        if best_solution > most_best:
            most_best = best_solution

        # 종료 조건
        # 1
        if generation > 1:
            if ll < 100:
                if (last_best == best_solution).all():
                    ll += 1
                else: 
                    ll = 0
            else: 
                break
        last_best = best_solution

        #2
        if time.time() - start_time > 60:
            break

    # print(f"optimal solution: {most_best}, {time.time()-start_time}")    
    # plt.clf()  # 그래프 초기화
    # plt.plot(range(1, generation+1), fit_list)
    # plt.xlabel('Generation')
    # plt.ylabel('Fitness')
    # plt.show()
        
    
