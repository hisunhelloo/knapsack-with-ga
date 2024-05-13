import random
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


def crossover(n, pop):
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


def offspring(n, value, weights, capacity, popsize, pop):
    offspring = crossover(n, pop)
    for i in range(popsize//2 - 1):
        childs = crossover(n, pop)
        offspring = np.append(offspring, childs, axis=0)

    for i in range(len(offspring)):
        offspring[i] = mutate(offspring[i], mutation_rate)
    
    # print(offspring)
    # print(len(offspring))
    
    for i in range(len(offspring)):
        if sum(offspring[i] * weights) > capacity:
            # print(f"{i}th: {offspring[i]}    {sum(offspring[i] * weights)} {sum(offspring[i] * weights) > capacity}")
            offspring[i] = np.array([0 for _ in range(n)])

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
        new_fitness.append(fitness[k])
    
    return np.array(new_population), np.array(new_fitness)


if __name__ == "__main__":

    # 주어진 값
    capacity = 24903
    n = 11
    weights = [6635, 4259, 3447, 8319, 9747, 7377, 9906, 22626, 8344, 642, 6428]
    value = [405, 480, 540, 645, 891, 52, 806, 363, 301, 662, 151]
    # opt = [1 1 1 0 1 0 0 0 0 1 0]
    weights = np.array(weights)
    value = np.array(value)

    popsize = 100
    mutation_rate = 0.2
    
    # 시작
    ## 첫 번째 population

    pop = population(n, value, weights, capacity, popsize)
    # pop = population2(n, value, weights, capacity, popsize)
    # print(pop)
    fitness = cal_fitness(pop, value)
    

    # 세대 반복
    generation = 0
    last_solution = []
    ll = 0
    while True:
        generation += 1
        child = offspring(n, value, weights, capacity, popsize, pop)
        # print(child)
        pop, fitness = new_pop(pop, child, value, popsize, weights)
        # print(pop)
        best_solution = pop[list(fitness).index(max(fitness))]
        last_solution.append(best_solution)
        print(f"{generation} generation's best solution: {best_solution}")


        # 종료 조건
        # 1
        if generation > 1:
            if ll < 100:
                if all(x1 == x2 for (x1, x2) in zip(last_solution[0], best_solution)):
                    ll += 1
                    last_solution.pop(0)
                else: 
                    ll = 0
            else: 
                break

        # 2
        sorted_population = sorted(pop, key = lambda x: fitness.all())
        last_index = int(0.9 * len(sorted_population)) - 1
        if (sorted_population[0] == sorted_population[last_index]).all():
            break
        
    print(f"optimal solution found({generation} generation): {best_solution}")
