## 폐기

import random
import numpy as np

class Knapsack:
    def __init__(self):
        self.popsize = 10
        self.mut_rate = 0.2
        self.values = []
        self.weights = []
        self.effect = []
        self.rank = []
        self.num_items = len(self.values)
        self.capacity = 40

    def initialize(self):
        self.effect = self.values / self.weights
        self.rank = np.argsort(self.effect)[::-1] + 1 
        pass  

    def properties(self, weights, values, capacity, population):
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity
        self.population = population
        self.initialize()      

value = np.array([2, 3, 4, 5, 6, 6])
weights = np.array([7, 5, 3, 7, 2, 1]) 

if __name__ == "__main__":
    problem1 = Knapsack()
