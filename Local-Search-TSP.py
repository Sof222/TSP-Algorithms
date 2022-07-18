import numpy as np
import re

import random 
import math

import matplotlib.pyplot as plt


def distance_between_nodes(n1, n2): #calculate the distance between two points
    x1 = n1[0]
    y1 = n1[1]

    x2 = n2[0]
    y2 = n2[1]

    return np.sqrt((np.square(int(x2) - int(x1)) + (np.square(int(y2) - int(y1)) )))


def distance_matrix(data): #create distance matrix which holds coordinates and their corresponding distances

    table = [[0 for i in range(len(data))] for j in range(len(data))] 

    for i in range(len(data)):
        for j in range(len(data)):
            table[i][j] = distance_between_nodes([data[i][1], data[i][2]], [data[j][1], data[j][2]])

    return table

def weight(m_d, ham_cycle): #calculate cost of cycle 
    return sum([m_d[i, j] for i, j in zip(ham_cycle, ham_cycle[1:] + [ham_cycle[0]])])



def probability(current, possible, temp):

    prob = math.exp(-(max(current-possible,1))/temp) #max added to account for 0 error

    return prob


def is_it_better(possible, current, d_m, temp): #decide if we want to chose this new point 

    current_weight = weight(np.asarray( d_m), current)
    possible_weight = weight(np.asarray( d_m), possible)

    if possible_weight < current_weight:
        return True

    else:
        if random.random() < probability(current_weight, possible_weight, temp):
            return True

    return False

def main():

    T0 = 1000000
    T_end = 0.0000001
    n = 280
    temp_decrease_rate = 0.5

    f = open('a280.txt','r')

    data = np.loadtxt('a280.txt', delimiter=',', skiprows=1, dtype=str)

    locations = []

    for d in data: #clean up text file data 
        d = d.strip()
        d = re.sub(' +', ' ', d)
        d = d.split()
        locations.append(d)


    dist_m = distance_matrix(locations)

    T = T0

    given_solution = []

    #randomly chosen given solution
    given_solution = [279, 278, 277, 276, 275, 274,268, 267, 266, 265, 133, 134, 135, 136, 137, 138,  273, 272, 271, 270, 269, 139, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 257, 258, 259, 260, 261, 262, 263, 264, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 202,  247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 201, 200, 199, 198, 197, 196, 195, 194, 193, 192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149, 148, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 279, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 81, 82, 83, 140, 141, 142, 143, 144, 145, 146, 147, 204, 203, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 98, 99, 100, 101, 102, 103, 104, 97, 96, 95, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 44, 43, 42, 41, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 232, 233, 234, 235]


    smallest_sol = weight(np.asarray(dist_m),given_solution)
    smallest_cycle = given_solution

    iterations = 0
    solutions_history=[]

    while T > T_end: #while stopping criteria not met

        possible = given_solution.copy()

        rand1 = random.randint(2, n - 1)
        rand2 = random.randint(0, n - 1)
        possible[rand2:(rand1 + rand2)] = reversed(possible[rand2:(rand1 + rand2)]) #switch random variables

        if is_it_better(possible, given_solution, dist_m, T): #if the new variation is better 

            given_solution = possible

            check_w = weight(np.asarray(dist_m),given_solution)

            if check_w  < smallest_sol: #update smallest cycle 
                
                smallest_sol = check_w 
                smallest_cycle = given_solution


        T = T /(1 + temp_decrease_rate) #decrease T value


        iterations += 1
        solutions_history.append(weight(np.asarray(dist_m),given_solution))


    print(smallest_sol)    
    print(smallest_cycle)

    
    plt.plot(np.linspace(1, iterations, num=iterations), solutions_history, '-')

    plt.xlabel('Number of generations')

    plt.ylabel('Minimum cost found')

    plt.show() #plot graph
    



main()


