import numpy as np
import linkedList as ll
import matplotlib.pyplot as plt
import random


# REMEBER DEL T = 0.5           SO 1 SECOND = 2 TIME STEPS


def reward(t,sigma = 2, mu = 40,rewardPresent = True): #sigma is ^2 right?
    if rewardPresent:
        return 0.5 * np.exp(-((t-mu)**2)/(2*sigma**2))
    else:
        return 0

def stim(t,i):
    retVAR = 0
    if t == 20:
        retVAR = 1
    return retVAR

def printA():
    plt.plot(np.arange(0,25,0.5),[reward(t) for t in np.arange(0,50,1)])
    plt.plot(np.arange(0,25,0.5),[stim(t,0) for t in np.arange(0,50,1)])
    plt.show()
#printA()
    
#build a big matric to store all 110 runs of the simulation each run has 50 time steps
#initialise the matrix with zeros

LEARNING_CYCLES = 110

data_V = np.zeros((LEARNING_CYCLES,50))
data_delV = np.zeros((LEARNING_CYCLES,50))
data_R = np.zeros((LEARNING_CYCLES,50))

W = np.zeros(25)

#using temporal diffrence learning to calculate the value of the state
def calculateValue(t,psi,value_la,value_cur, S, rewardState,alpha=0.2,gamma=1):
    global W
    delta = (reward(t,rewardPresent=rewardState) + gamma * value_cur - value_la)
    print(delta)
    return W + alpha * delta *S#*psi

def populateData():

    global W
    for i in range(0,LEARNING_CYCLES,1):
        
        T_mem = ll.FixedSizeStack(25)
        T_mem.fill_with_zeros()
        value_last = np.zeros(1)

        #reward state true only 50% of the time
        p = 1
        rewardState = False
        if random.random() < p:
            rewardState = True

        print(rewardState)
        for t in range(50):

            T_mem.push(stim(t,i))
            value_cur = W @ T_mem.to_numpy().reshape(25,1)
            
        
            alpha = 0.01
            gamma = 1
            W = calculateValue(t, T_mem.to_numpy(), value_last ,value_cur,T_mem.containsStim(), rewardState, alpha, gamma)



            data_V[i][t] = value_cur
            data_delV[i][t] = gamma * value_cur - value_last
            data_R[i][t] = reward(t) - reward(t-1) + alpha * value_cur - value_last
            value_last = value_cur

populateData()

def plotData():
    for numbers in range(0,LEARNING_CYCLES,10):  #LEARNING_CYCLES
        plt.plot(np.arange(0,25,0.5),data_V[numbers])
        #plt.plot(np.arange(0,25,0.5),data_delV[numbers])
        #plt.plot(np.arange(0,25,0.5),data_R[numbers])
    plt.show()

plotData()