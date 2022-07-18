import array
import random
import matplotlib.pyplot as plt

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import re


def distance_between_nodes(n1, n2): #calculate the distance bewteen two points
    x1 = n1[0]
    y1 = n1[1]

    x2 = n2[0]
    y2 = n2[1]

    return numpy.sqrt((numpy.square(int(x2) - int(x1)) + (numpy.square(int(y2) - int(y1)) )))

def distance_matrix(data): #generate distance matrix for coordinates provided 

    table = [[0 for i in range(len(data))] for j in range(len(data))] 

    for i in range(len(data)):
        for j in range(len(data)):
            table[i][j] = distance_between_nodes([data[i][1], data[i][2]], [data[j][1], data[j][2]])

    return table

f = open('a280.txt','r')
data = numpy.loadtxt('a280.txt', delimiter=',', skiprows=1, dtype=str)

locations = []

for d in data: #reformat the data 
    d = d.strip()
    d = re.sub(' +', ' ', d)
    d = d.split()
    locations.append(d)

distance_map = distance_matrix(locations)

IND_SIZE = 280 #size of the number of coordinates 

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(ind): #evaluate based off cost of cycle
    distance = distance_map[ind[-1]][ind[0]]

    for g1, g2 in zip(ind[0:-1], ind[1:]):
        distance += distance_map[g1][g2]
    return distance,

toolbox.register("mate", tools.cxPartialyMatched) #crossover
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2) #mutation operation
toolbox.register("select", tools.selTournament, tournsize=4) #selection 
toolbox.register("evaluate", evaluate) #evalue method

def main():
    random.seed(69)

    pop = toolbox.population(n=100)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values) #stats to show 
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    p, logbook = algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 10, stats=stats, 
                        halloffame=hof)

    gen = logbook.select("gen")

    min_v = logbook.select("min")

    plt.plot(gen, min_v, 'o')

    plt.xlabel('Number of generations')

    plt.ylabel('Minimum cost found')

    plt.show() #plot graph

    return pop, stats, hof

if __name__ == "__main__":
    main()
