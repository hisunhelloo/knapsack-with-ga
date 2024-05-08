import random
import numpy as np

# 주어진 값
value = np.array([2, 3, 4, 5, 6, 6])
weights = np.array([7, 5, 3, 7, 2, 1])
num_items = len(value)
capacity = 15
popsize = 10
mutation_rate = 0.2


def effect_rank(value, weights, num_items):
    # effect 계산
    effect = value / weights

    # 순위 매기기
    rank = np.argsort(effect)[::-1] + 1  # 큰 값 순으로 순위 매기기
    print(rank)

    # 배열 정렬하기
    new_value = np.zeros(num_items, dtype=int)
    new_weights = np.zeros(num_items, dtype=int)
    for i in range(num_items):
        new_value[rank[i]-1] = value[i]
        new_weights[rank[i]-1] = weights[i]
    
    return new_value, new_weights, rank

def generate_chromosome(chromosome, new_weights, index, capacity, pop):

    if index == len(chromosome):

        if sum(chromosome*new_weights) <= capacity:
            pop.append(chromosome.copy())
        return
    
    # 추가
    chromosome[index] = 1
    generate_chromosome(chromosome, new_weights, index + 1, capacity, pop)
    
    # 제거
    chromosome[index] = 0
    generate_chromosome(chromosome, new_weights, index + 1, capacity, pop)

def population(new_value, new_weights, capacity, popsize):

    # 새로운 pop 생성
    pop = []
    chromosome = np.zeros(num_items, dtype=int)
    generate_chromosome(chromosome, new_weights, 0, capacity, pop)

    return pop[:popsize]
    

def cal_fitness(pop, new_value):
    fitness_score = [sum(pop[i]*new_value) for i in range(len(pop))]
    return fitness_score

def is_over_capacity(pop, new_weights, capacity):
    for i in range(len(pop)):
        if (sum(pop[i]*new_weights) > capacity).all():
            pop.pop(i)
    return

def tournament(fitness_score, fitness_scores):
    pop_size = len(fitness_score)
    n = random.randint(0, pop_size-1)
    for _ in range(2):
        m = random.randint(0, pop_size-1)
        if (fitness_score[n] > fitness_score[m]).all():
            n = m
    
    return fitness_score[n]

def crossover(parent1, parent2):
    split_index = random.randint(1, len(parent1)-1)
    child1 = np.concatenate((parent1[:split_index], parent2[split_index:]),axis=0)
    child2 = np.concatenate((parent2[:split_index], parent1[split_index:]),axis=0)
    return child1, child2

def mutate(offspring, mutation_rate):
    if random.random() < mutation_rate:
        n = random.randint(0, len(offspring)-1)
        offspring[n] = 1 - offspring[n]
    return offspring


def offspring(population, new_value, new_weights, capacity):
    population_size = len(population)
    fitness_score = cal_fitness(population, new_value)
    offspring = []
    for i in range(population_size//2):
        parent1 = tournament(population, fitness_score)#check
        parent2 = tournament(population, fitness_score)
        child1, child2 = crossover(parent1, parent2)
        offspring.append([child1, child2])
    
    for i in range(len(offspring)):
        offspring[i] = mutate(offspring[i], mutation_rate)
    
    is_over_capacity(offspring, new_weights, capacity)

    return offspring

def new_pop(population, offspring, new_value, popsize):
    population = np.array(population)
    offspring = np.array(population)
    temp = np.concatenate((population, offspring), axis=0)

    fit_factor = cal_fitness(temp, new_value)
    # print(len(temp), len(fit_factor))


    total_fitness = sum(fit_factor)
    fit_factor = np.array(fit_factor) / total_fitness
    # print(fit_factor)

    new_population = []
    new_fitness = []
    for i in range(popsize):
        k = random.choices(range(len(temp)), weights = fit_factor)
        new_population.append(temp[k][0])
        new_fitness.append(fit_factor[k])
    return new_population, new_fitness

    
def best_solution(population, fitness):
    print(f"best solution: {population[fitness.index(max(fitness))]}")

def is_end(population, fitness):
    sorted_population = sorted(population, key = lambda x: fitness)
    sorted_fitness = fitness.sort(reverse =True)
    last_index = int(0.9 * len(sorted_population)) - 1
    
    if (sorted_population[0] == sorted_population[last_index]).all():
        return True
    return False

if __name__ == "__main__":
    
    ## 1 (성공, 0)
    # value = np.array([2, 3, 4, 5, 6, 6])
    # weights = np.array([7, 5, 3, 7, 2, 1])
    # capacity = 15

    ## 2 (성공, 5)
    # value = np.array([2,7,9,3,5,9,3,6,2,1,7])
    # weights = np.array([4, 3, 4, 7, 2, 8, 4, 8, 9, 1, 5])
    # capacity = 20

    ## 3 (성공, 5)
    # value = np.array([2,7,9,3,5,9,3,6,2,1,7])
    # weights = np.array([4, 3, 4, 7, 2, 8, 4, 8, 9, 1, 5])
    # capacity = 30

    ## 3 (성공, 6)
    # value = np.array([random.randint(1,15) for i in range(20)])
    # weights = np.array([random.randint(1,15) for i in range(20)])
    # capacity = 40

    ## 4 (성공, 3)
    # value = np.array([random.randint(1,15) for i in range(20)])
    # weights = np.array([random.randint(1,15) for i in range(20)])
    # capacity = 20


    # 주어진 값
    num_items = len(value)
    popsize = 10
    mutation_rate = 0.2

    # 시작
    ## 첫 번째 population
    new_value, new_weights, rank = effect_rank(value, weights, num_items)
    pop = population(new_value, new_weights, capacity, popsize)#check
    print(pop)
    
    fitness = cal_fitness(pop, new_value)#check
    

    # 세대 반복
    generation = 0
    while is_end(pop, fitness) != True:
        generation += 1
        child = offspring(pop, new_value, new_weights, capacity)
        print(child)#check
        exit(0)
        pop, fitness = new_pop(pop, child, new_value,popsize)
        print(pop)

        print(f"{generation} generation's best solution: {pop[fitness.index(max(fitness))]}")

    print(f"optimal solution found({generation} generation): {pop[fitness.index(max(fitness))]}")
