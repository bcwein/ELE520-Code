import numpy as np
import math
import scipy.linalg as la
import matplotlib.pyplot as plt
############################################### WRITE FROM LINE 33#####################################################
def areSame(A,B):
    n = 2
    for i in range(n):
        for j in range(n):
            if (A[i][j] != B[i][j]):
                return 0
    return 1    


def decBorder( sigma1, sigma2, my1, my2, pw1, pw2):
    x1 = [-20, 20]
    if areSame(sigma1,sigma2):
        print("case 2")
        
         
        w10 = (-1/2)* np.dot(np.dot(my1, np.linalg.inv(sigma1)), np.transpose(my1)) + math.log(pw1)
        w20 = (-1/2)* np.dot(np.dot(my2, np.linalg.inv(sigma1)), np.transpose(my2)) + math.log(pw2)
        
        w1 = np.linalg.inv(sigma1) *my1
        w2 = np.linalg.inv(sigma1) *my2

        
        w1 = np.dot(np.linalg.inv(sigma1) ,np.transpose(my1))
        
        w2 = np.dot(np.linalg.inv(sigma2) ,np.transpose(my2))
        
        #w2[0][0], "x1 + ",w1[1][0]

        print("w1:", w1," w2:", w2)
        print("w10: ", w10, "w20: ", w20)
        coordinate1 =(w1[0][0]*x1[0]- w2[0][0]*x1[0] +w10 - w20)/(-w1[1][0]+w2[1][0])
        coordinate2 =(w1[0][0]*x1[1]- w2[0][0]*x1[1] +w10 - w20)/(-w1[1][0]+w2[1][0])
        print("Decision border: ")
        dummy = (w10 - w20)/(-w1[1][0]+w2[1][0])
        
        print("x2 = ",(w1[0][0]- w2[0][0])/(-w1[1][0]+w2[1][0]),"x1 + ",dummy[0][0])
        
        return [[x1[0],coordinate1], [x1[1],coordinate2]]
    else:
        print("case 3")
           
        w10 = (-1/2)* np.dot(np.dot(my1, np.linalg.inv(sigma1)), np.transpose(my1)) - (1/2) * math.log(np.linalg.det(sigma1)) + math.log(pw1)
        w20 = (-1/2)* np.dot(np.dot(my2, np.linalg.inv(sigma2)), np.transpose(my2))- (1/2) * math.log(np.linalg.det(sigma2)) + math.log(pw2)
        
        w1 = np.linalg.inv(sigma1) *my1
        w2 = np.linalg.inv(sigma2) *my2
        W1 = (-1/2) * np.linalg.inv(sigma1)
        W2 = (-1/2) * np.linalg.inv(sigma2)
        print("w1:", w1," w2:", w2)
        print("w10: ", w10, "w20: ", w20)
        coordinate1 =(w1[0,0]*x1[0]- w2[0,0]*x1[0] +w10 - w20)/(-w1[1,1]+w2[1,1])
        coordinate2 =(w1[0,0]*x1[1]- w2[0,0]*x1[1] +w10 - w20)/(-w1[1,1]+w2[1,1])


        w1x12 = W1[0,0]
        w1x1x2 = W1[0,1] + W1[1,0]
        w1x22 = W1[1,1]

        w2x12 = W2[0,0]
        w2x1x2 = W2[0,1] + W2[1,0]
        w2x22 = W2[1,1]

        #first part
        x12 = w1x12 - w2x12 
        x22 = w1x22 - w2x22
        x1x2 = w1x1x2 - w2x1x2

        #second part
        
        w1 = np.dot(np.linalg.inv(sigma1) ,np.transpose(my1))
        
        w2 = np.dot(np.linalg.inv(sigma2) ,np.transpose(my2))
       
        dummy = (w10 - w20)
        print("Decision border= ",x12,"x1^2 + ", x22 ,"x2^2 + ",x1x2,"x1x2 + ",w1[0][0]- w2[0][0], "x1 + ",w1[1][0]-w2[1][0],"x2 + ",dummy[0][0]   )
        print("Decision border copy to desmos: ",x12,"x^2 + ", x22 ,"y^2 + ",x1x2,"xy + ",w1[0][0]- w2[0][0], "x + ",w1[1][0]-w2[1][0],"y + ",dummy[0][0], " = 0"  )
        
                
        x = np.linspace(-50, 50, 500)
        y = np.linspace(-50, 50, 500)
        X, Y = np.meshgrid(x, y)
        F = x12*X**2 + x22*Y**2   + x1x2*X*Y + (w1[0][0]- w2[0][0])*X + (w1[1][0]-w2[1][0])*Y + dummy[0][0]

        fig,ax = plt.subplots()
        ax.contour(X, Y, F, levels=[0]) # take level set corresponding to 0
        
                
      
        return 0


####################################################### write here ############################################################################

sigma1 = np.array([[0.5,0],[0,0.5]])  # write sigma 1
sigma2 = np.array([[2.045,0.3],[0.3,2]])  # write sigma 2

my1 = np.array([[1,1]])         # write my1
my2 = np.array([[4,4]])           # write my2
pw1 = 0.5                         # write pw1
pw2 = 1-pw1
unknowndot = [2.75,2.75] # write if need to classify

# run python file in terminal -> get decision border and countour line as popup + ligning i terminal: x2 = aX1 + b
















decborder = decBorder(sigma1, sigma2, my1, my2, pw1, pw2)
sig = np.array([[0.5,0],[0,0.5]])
eigvals, eigen_vectors = la.eig(sigma1)

eigvals2, eigen_vectors2 = la.eig(sigma2)

eigvals = eigvals.real

eigvals2 = eigvals2.real
eig_vec1 = eigen_vectors[:,0]
eig_vec2 = eigen_vectors[:,1]

eig_vec3 = eigen_vectors2[:,0]
eig_vec4 = eigen_vectors2[:,1]
print("Eigenvalues class 1: ", eigvals," class 2: ", eigvals2)
print("eigenvectors clas 1: ", eig_vec1," class 2: ", eig_vec2)
# grid space
plt.ylim(-15, 7)
plt.xlim(-15, 7)
#plt.show()
# This line below plots the 2d points
plt.scatter(my1[0,0], my1[0,1])
plt.scatter(my2[0,0], my2[0,1])
plt.text(my1[0,0], my1[0,1], " class w1")
plt.text(my2[0,0], my2[0,1], " class w2")
#unknown dot
plt.scatter(unknowndot[0], unknowndot[1])
plt.text(unknowndot[0], unknowndot[1], " unknown class")
if decborder != 0:
    xvals = [decborder[0][0], decborder[1][0]]
    yvals = [decborder[0][1][0][0],decborder[1][1][0][0]]
    plt.plot(xvals, yvals)

scaledEigenVector1 = my1[0] + math.sqrt(eigvals[0]) * eig_vec1
xvals = [my1[0,0], scaledEigenVector1[0]]
yvals = [my1[0,1], scaledEigenVector1[1]]
plt.plot(xvals, yvals)
scaledEigenVector1 = my1[0] - math.sqrt(eigvals[0]) * eig_vec1
xvals = [my1[0,0], scaledEigenVector1[0]]
yvals = [my1[0,1], scaledEigenVector1[1]]
plt.plot(xvals, yvals)


#print("Scaled eigen vectors: ", scaledEigenVector1)
scaledEigenVector2 = my1[0] + math.sqrt(eigvals[1]) * eig_vec2
xvals1 = [my1[0,0], scaledEigenVector2[0]]
yvals1 = [my1[0,1], scaledEigenVector2[1]]
plt.plot(xvals1, yvals1)
scaledEigenVector2 = my1[0] - math.sqrt(eigvals[1]) * eig_vec2
xvals1 = [my1[0,0], scaledEigenVector2[0]]
yvals1 = [my1[0,1], scaledEigenVector2[1]]
plt.plot(xvals1, yvals1)
#print("Scaled eigen vectors: ", scaledEigenVector2)


scaledEigenVector1 = my2[0] + math.sqrt(eigvals2[0]) * eig_vec3
xvals = [my2[0,0], scaledEigenVector1[0]]
yvals = [my2[0,1], scaledEigenVector1[1]]
plt.plot(xvals, yvals)
scaledEigenVector1 = my2[0] - math.sqrt(eigvals2[0]) * eig_vec3
xvals = [my2[0,0], scaledEigenVector1[0]]
yvals = [my2[0,1], scaledEigenVector1[1]]
plt.plot(xvals, yvals)

scaledEigenVector2 = my2[0] + math.sqrt(eigvals2[1]) * eig_vec4
xvals1 = [my2[0,0], scaledEigenVector2[0]]
yvals1 = [my2[0,1], scaledEigenVector2[1]]
plt.plot(xvals1, yvals1)
scaledEigenVector2 = my2[0] - math.sqrt(eigvals2[1]) * eig_vec4
xvals1 = [my2[0,0], scaledEigenVector2[0]]
yvals1 = [my2[0,1], scaledEigenVector2[1]]
plt.plot(xvals1, yvals1)
#print("Scaled eigen vectors w2: ", scaledEigenVector1)
#print("Scaled eigen vectors w1: ", scaledEigenVector2)
ymin = min([my1[0,1],my2[0,1]])

ymax = max(my1[0,1],my2[0,1])

xmin = min(my1[0,0],my2[0,0])
xmax = max(my1[0,0],my2[0,0])
plt.ylim(ymin-3, ymax +3)
plt.xlim(xmin - 3, xmax + 3)

plt.grid(color='lightgray',linestyle='--')

plt.show() 