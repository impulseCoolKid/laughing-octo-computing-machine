import numpy as np
import matplotlib.pyplot as plt
import random


# REMEBER DEL T = 0.5           SO 1 SECOND = 2 TIME STEPS


def reward(t,sigma = 2, mu = 40,rewardPresent = True): #sigma is ^2 right?
    if rewardPresent:
        return 0.5 * np.exp(-((t-mu)**2)/(2*sigma**2))
    else:
        return 0

def stim(t):
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

LEARNING_CYCLES = 1000

data_V = np.zeros((LEARNING_CYCLES,50))
data_delV = np.zeros((LEARNING_CYCLES,50))
data_R = np.zeros((LEARNING_CYCLES,50))

#UNrewarded
UNrewardeddata_V = np.zeros((LEARNING_CYCLES,50))
UNrewardeddata_delV = np.zeros((LEARNING_CYCLES,50))
UNrewardeddata_R = np.zeros((LEARNING_CYCLES,50))

#REwarded 
REwardeddata_V = np.zeros((LEARNING_CYCLES,50))
REwardedData_delV = np.zeros((LEARNING_CYCLES,50))
REwardedData_R = np.zeros((LEARNING_CYCLES,50))



#using temporal diffrence learning to calculate the value of the state

def populateData(p=1):
    W = np.zeros(25)
    print(W)
    for i in range(0,LEARNING_CYCLES,1):
        
        #reward state true only 50% of the time
        
        rewardState = False
        if random.random() < p:
            rewardState = True
        

        T_mem = np.zeros(25)
        epsilon = 0.01
        gamma = 1
        for t in range(50):
            psi = np.cumsum(T_mem)
            value_cur = np.dot(W, psi)
            
            
            T_mem_fut = np.concatenate(([stim(t+1)], T_mem [:-1]))
            psi_fut = np.cumsum(T_mem_fut)
            valuefuture = np.dot(W, psi_fut) #if value is location dep it will make it matter more
            
            delta = reward(t,rewardPresent=rewardState) + gamma* valuefuture - value_cur

            W = W + epsilon * delta * psi

            data_V[i][t] = value_cur
            data_delV[i][t] = gamma * valuefuture - value_cur
            data_R[i][t] = reward(t-1)  + gamma * valuefuture - value_cur

            if rewardState and i > 900:
                REwardeddata_V[i][t] = value_cur
                REwardedData_delV[i][t] = gamma * valuefuture - value_cur
                REwardedData_R[i][t] = reward(t-1)  + gamma * valuefuture - value_cur
            elif not rewardState and i > 900:
                UNrewardeddata_V[i][t] = value_cur
                UNrewardeddata_delV[i][t] = gamma * valuefuture - value_cur
                UNrewardeddata_R[i][t] = reward(t-1)  + gamma * valuefuture - value_cur

            T_mem = T_mem_fut
            

#populateData()

colors = plt.cm.jet(np.linspace(0, 1, 100))

def plotData(data_V=data_V,data_delV=data_delV,data_R=data_R):

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for numbers in range(900,LEARNING_CYCLES,1):  #LEARNING_CYCLES
        axs[0].plot(np.arange(0,25,0.5),data_V[numbers],color=colors[numbers-900])
        axs[1].plot(np.arange(0,25,0.5),data_delV[numbers],color=colors[numbers-900])
        axs[2].plot(np.arange(0,25,0.5),data_R[numbers],color=colors[numbers-900])
        #plt.legend(["Run 1","Run 2","Run 3","Run 4","Run 5","Run 6","Run 7","Run 8","Run 9","Run 10","Run 11"])

    axs[0].set_ylabel("V(t)")
    axs[0].set_title("Value Estimates")
    axs[1].set_ylabel("ΔV(t)")
    axs[1].set_title("Temporal Difference in Value")
    axs[2].set_ylabel("δ(t)")
    axs[2].set_title("TD Learning Error")
    axs[2].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()

'''populateData(p=1)
plotData()'''

"""plotData(data_V=REwardeddata_V,data_delV=REwardedData_delV,data_R=REwardedData_R)
plotData(data_V=UNrewardeddata_V,data_delV=UNrewardeddata_delV,data_R=UNrewardeddata_R)"""

def dopamine(x):
    a = 6
    b = 6
    x_star =0.27
    if x<0:
        return x/a
    elif 0<=x<=x_star:
        return x
    elif x_star<=x:
        return x_star + (x-x_star)/b
    

def adverageDopamineSignal(): #Q6
    dataCut = data_R[-100:, :]
    data =np.mean(dataCut, axis=0) 
    print(data.shape)
    for i in range(50):
        data[i] = dopamine(data[i])

    plt.plot(np.arange(0,25,0.5),data)
    return data

'''populateData(p=0.5)
adverageDopamineSignal()
plt.show()'''

def dopamineForAllTrials(): #Q7
    legend = []
    for numbers in np.arange(0, 1.25, 0.25):
        populateData(p=numbers)
        legend.append(f"p = {numbers}")
        adverageDopamineSignal()
    plt.legend(legend)
    plt.show()

dopamineForAllTrials()

def stimAndRewardDopamine(): # Q 8
    atReward = []
    atStim = []
    num = []
    for numbers in np.arange(0, 1.25, 0.25):
        num.append(numbers)
        populateData(p=numbers)
        data = adverageDopamineSignal()
        #cut at 13 -> 26
        dataStim = data[:26]
        dataReward = data[26:]
        atReward.append(np.max(dataReward))
        atStim.append(np.max(dataStim))
    plt.clf()
    plt.plot(num,atReward)
    plt.plot(num,atStim)
    plt.legend(["At Reward"," At Stimulus"])
    plt.xlabel("Probability of Reward")
    plt.ylabel("Dopamine Signal")
    plt.show()

#stimAndRewardDopamine()