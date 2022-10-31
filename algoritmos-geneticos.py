import random
import pandas as pd
import numpy as np

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def f(x,y):
    return np.sin(x)*np.exp(pow(1-np.cos(y), 2)) + np.cos(y)*np.exp(pow(1-np.sin(x), 2)) + pow((x-y),2)

def plot(points):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig = plt.figure()
    ax = mplot3d.Axes3D(fig=fig, computed_zorder=False)
    
    # Plot graph
    ax.plot_surface(X, Y, Z, rstride=2, cstride=2,
                    cmap='summer', edgecolor='none', zorder=0.5)
    ax.set_title('bird')
    
    # Plot points
    xdata = np.array(points).T[0]
    ydata = np.array(points).T[1]
    zdata = f(xdata, ydata)
    ax.scatter3D(xdata, ydata, zdata, c='black', linewidths=1, zorder=1)
    
    fig.show()

def linear_rank(list):

    # Variables to make df
    results = []
    x=[]
    y=[]
    ranking=[]
    cumulative_sum=[]
    for individual in list:
        # Save x, y and f(x,y)
        results.append(f(individual[0],individual[1]))
        x.append(individual[0])
        y.append(individual[1])
        # Dummy values -1
        ranking.append(-1)
        cumulative_sum.append(-1)

    # Create and sort array by results
    data = pd.DataFrame(np.array(np.array([x,y,results,ranking,cumulative_sum]).T), columns=['x','y','result','ranking', 'soma'])
    data = data.sort_values(by=['result'], ascending=True, ignore_index=True)

    # Rank all from 0 to 100
    data['ranking'][0] = 100
    data['ranking'][len(data) - 1] = 0
    for i in data.index:
        if data['ranking'][i] == -1:
            data['ranking'][i] = 100*((len(data) - i - 1)/(len(data) - 1))
    return data


# Roll new (n) individuals, given that better ranked ones
#have more chance to be chosen. Can have duplicates.
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

    for i in range(0, int(len(selected)/2)):
        
        # Checks if pair will breed
        if random.uniform(0,1) > breeding_rate:
            continue

        index = 2*i

        alpha = random.uniform(0,1)
        children1 = [selected[index][0]*alpha + (1-alpha)*selected[index+1][0], 
                     selected[index][1]*alpha + (1-alpha)*selected[index+1][1]]
        children2 = [selected[index+1][0]*alpha + (1-alpha)*selected[index][0],
                     selected[index+1][1]*alpha + (1-alpha)*selected[index][1]]
        selected[index] = children1
        selected[index+1] = children2


def mutation(selected, mutation_rate):
    pass

if __name__ == "__main__":

    size = 100
    breeding_rate = 0.7
    mutation_rate = 0.001
    iterations = 100

    population = []

    for x in range(0,size):
        population.append([random.uniform(-10, 10),random.uniform(-10, 10)])
    
    for it in range(0, iterations):
    
        ranking = linear_rank(population)

        population = roll(ranking, size)

        arithmetic_crossover(population, breeding_rate)

        mutation(population, mutation_rate)

    plot(population)

    print('test')
    