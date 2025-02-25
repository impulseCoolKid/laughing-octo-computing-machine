import numpy as np
import matplotlib.pyplot as plt


# REMEBER DEL T = 0.5           SO 1 SECOND = 2 TIME STEPS


def reward(t,sigma = 2, mu = 40): #sigma is ^2 right?
    return 0.5 * np.exp(-((t-mu)**2)/(2*sigma**2))

def stim(t):
    retVAR = 0
    if t == 20:
        retVAR = 1
    return retVAR

def printA():
    plt.plot(np.arange(0,25,0.5),[reward(t) for t in np.arange(0,50,1)])
    plt.plot(np.arange(0,25,0.5),[stim(t) for t in np.arange(0,50,1)])
    plt.show()
#printA()
    
#build a big matric to store all 110 runs of the simulation each run has 50 time steps
#initialise the matrix with zeros

LEARNING_CYCLES = 210

data_V = np.zeros((LEARNING_CYCLES,50))
data_delV = np.zeros((LEARNING_CYCLES,50))
data_R = np.zeros((LEARNING_CYCLES,50))

W = np.zeros(25)
print(W)
#using temporal diffrence learning to calculate the value of the state

def populateData():

    global W
    for i in range(0,LEARNING_CYCLES,1):
        
        T_mem = np.zeros(25)
        epsilon = 0.2
        gamma = 1

        for t in range(50):
            
            value_cur = np.dot(W, T_mem)
            print(value_cur)
            
            T_mem_fut = np.concatenate(([stim(t+1)], T_mem [:-1]))
            valuefuture = np.dot(W, T_mem_fut) #if value is location dep it will make it matter more
            
            delta = reward(t) + gamma* valuefuture - value_cur

            W = W + epsilon * delta * T_mem

            data_V[i][t] = value_cur
            data_delV[i][t] = gamma * valuefuture - value_cur
            data_R[i][t] = reward(t-1)  + gamma * valuefuture - value_cur
            T_mem = T_mem_fut
            

populateData()

colors = plt.cm.jet(np.linspace(0, 1, LEARNING_CYCLES))
def plotData():
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for numbers in range(0,LEARNING_CYCLES,10):  #LEARNING_CYCLES
        axs[0].plot(np.arange(0,25,0.5),data_V[numbers],color=colors[numbers])
        
        axs[1].plot(np.arange(0,25,0.5),data_delV[numbers],color=colors[numbers])
        axs[2].plot(np.arange(0,25,0.5),data_R[numbers],color=colors[numbers])
       

    axs[0].set_ylabel("V(t)")
    axs[0].set_title("Value Estimates")
    axs[1].set_ylabel("ΔV(t)")
    axs[1].set_title("Temporal Difference in Value")
    axs[2].set_ylabel("δ(t)")
    axs[2].set_title("TD Learning Error")
    axs[2].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()

plotData()