import numpy as np
import math


def LMS(theta, tHold, Xs, Ys, rate, i=1,  k=0):
    notComplete = True
    flagList = {}
    length = len(Xs)

    for index in range(0, length):  ##Set all Xs to false
        flagList[index] = False
    flagTrues = 0

    while notComplete:
       
        raten = rate/i  
        xi = np.array(Xs[k]) 
        yi = Ys[k]
        k = k+1 % length
        if k == length:
            print("Reached end of list, trying again")
            k = 0
            for j in range(0, len(flagList)):
                flagList[j] = False
            flagTrues = 0
        xiT = xi.reshape(-1, 1)
        theta2 = raten*(yi-(np.dot(theta[0], xiT)))*xiT
        summen = math.sqrt(theta2[0]**2 + theta2[1]**2 + theta2[2]**2)
        if summen > tHold and flagList[k] == False:
            i = i + 1
            print("------------------------")
            print(i-1, k,  theta[0], raten, summen, theta2.transpose())
            print("--------------------------")
            x1 = xiT
            y1 = yi
            tobeTransposed = raten*(y1-(np.dot(theta[0], x1)))*x1
            # rate*(y1-(np.dot(theta[0], x1)))*x1
            theta = theta + tobeTransposed.transpose()
        elif summen < tHold: 
            flagList[k] = True #Sette denne Xen til true
            flagTrues += 1
        if flagTrues == length: 
            notComplete = False
            print("Process complete")
            print("==============================")
            print("the final theta ", theta[0])
            print("gang denne thetaen med (x1 x2 1) eller (1 x1 x2) som oppgitt i oppgavebeskrivelsen.")
            print("===============================")
            break

################# WRITE HERE ##########################################
theta = np.array([(1, 1, 1)])  # initial starting point theta
tHold = 0.5 # Threshold som float
rate = 1.0  # learning rate også kalt "nm"

# Fill inn x = (x1 x2 1) eller x = (1 x1 x2)
Xs = np.array([[1, 2, 1], [2, 0, 1], [3, 1, 1], [2, 3, 1]])
Ys = [1, 1, -1, -1]  # 1 for class 1, -1 for class 2

LMS(theta, tHold, Xs, Ys, rate)
