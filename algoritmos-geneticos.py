
import math
import random
import pandas as pd
import numpy as np

def f(x,y):
    return math.sin(x)*math.exp(pow(1-math.cos(y), 2)) + math.cos(y)*math.exp(pow(1-math.sin(x), 2)) + pow((x-y),2)

def linear_rank(list):
    results = []
    x=[]
    y=[]
    ranking=[]
    cumulative_sum=[]
    for individual in list:
        results.append(f(individual[0],individual[1]))
        x.append(individual[0])
        y.append(individual[1])
        ranking.append(-1)
        cumulative_sum.append(-1)

    data = pd.DataFrame(np.array(np.array([x,y,results,ranking,cumulative_sum]).T), columns=['x','y','result','ranking', 'soma'])
    data = data.sort_values(by=['result'], ascending=True, ignore_index=True)

    data['ranking'][0] = 100
    data['ranking'][len(data) - 1] = 0
    for i in data.index:
        if data['ranking'][i] == -1:
            data['ranking'][i] = 100*((len(data) - i - 1)/(len(data) - 1))
    return data

def roll(data, size):

    selecteds = []

    cumulative_sum = 0
    for i,line in enumerate(data.values):
        cumulative_sum += line[3]
        data['soma'][i] = cumulative_sum

    for x in range(size):
        selected = random.uniform(0, cumulative_sum)
        for line in data.values:
            if selected < line[4]:
                selecteds.append([line[0], line[1]])
                break

    return selecteds


def arithmetic_crossover(selected, breeding_rate):
    pass


if __name__ == "__main__":

    size = 100
    breeding_rate = 0.7

    population = []

    for x in range(0,size):
        population.append([random.uniform(-10, 10),random.uniform(-10, 10)])
    
    ranking = linear_rank(population)

    selected = roll(ranking, size)

    # crossover aritmetico, casais em ordem. Gerar um valor entre 0 e 1. 
    # Tem que ver se vai cruzar ou não (precisa de uma taxa de cruzamento em torno de 0.7), perguntando uma vez por casal. 
    # alpha é um valor aleatorio entre 0 e 1, fazendo em x e y. Gerar os filhos e substituir os pais.

    arithmetic_crossover(selected, breeding_rate)

    print('test')
    