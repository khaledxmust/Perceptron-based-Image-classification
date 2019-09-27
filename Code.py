#Importing
import os
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix
np.set_printoptions(threshold=np.inf) #Printing Threshold

#%% Part One - Training

Train_Path='C:\\Users\\khled\\Dropbox\\Masters\\Courses\\xDone\\Machine Learning\\Assigments\\Assignment 1\\Train'
os.chdir(Train_Path)
Training_Labels = np.loadtxt('Training Labels.txt')
files=os.listdir(Train_Path)
files.pop()
files = sorted(files,key=lambda x: int(os.path.splitext(x)[0]))

Training=[] #Importing and Reshaping
for i in files:    
    img=misc.imread(i)
    type(img)
    img.shape
    img=img.reshape(784,)
    img=np.append(img,1)
    Training.append(img)
#print(Classes[0].shape)
#print(len(Classes))
#print(len(img))

W=[] #Creating a Weight Vectors
w =np.zeros((len(img),1))
w[0]=1
for i in range(0,10):
    W.append(w)
#print('weight vector')
#print(list(W))
#print(len(W))
#print(W[0])

T=[] #Initialize Target Classes
for j in range (0,10):
    t=np.zeros(len(Training))
    for x in range(0,10):
        for i in range(2400):
            if Training_Labels[i]==j:
                t[i]=1
            else:
                t[i]=-1
    T.append(t)        
#print(T[0])
#print(len(T))

eta =[1, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6, 10**-7, 10**-8, 10**-9] #Training Rates
#print('learning rate η')
#print(n)

 #%% Error Handling and Updating Weight vectors (Main algorithm)

for x in range(0,2000): #Takes Avarage of 4 Days to "Run" (Needed for Conversion)
    for n in range(0,10): # Main algorithm '45 Seconds'
        for i in range(0,len(Training)):
            Y = np.dot(W[n].T, Training[i])*T[n][i]
            if (Y < 1).any():
                W[n] = W[n] +eta[0]*Training[i]*T[n][i]
#print(W[0])

#%% Part Two - Testing

Test_Path='C:\\Users\\khled\\Dropbox\\Masters\\Courses\\xDone\\Machine Learning\\Assigments\\Assignment 1\\Test'
os.chdir(Test_Path)
Test_Labels = np.loadtxt('Test Labels.txt')
files=os.listdir(Test_Path)
files.pop()
files = sorted(files,key=lambda x: int(os.path.splitext(x)[0]))

Testing=[] #Importing and Reshaping
for i in files:    
    img=misc.imread(i)
    type(img)
    img.shape
    img=img.reshape(784,)
    img=np.append(img,1)
    Testing.append(img)
#print(Testing[0].shape)
#print(len(Testing))
#print(len(img))

Weighted_imgs =  [] #Multiplying each Test_img with all Weights
for x in range(0,200):
    xWeighted = []
    for i in range(0,10):
        t = (np.dot(W[i].T,Testing[x]))
        xWeighted.append(t)
    Weighted_imgs.append(xWeighted)
#print(Weighted_imgs.keys())
    
Maxv = [] #Max Weight Classification
xMaxv = []
for i in range(0,200):
    Z = Weighted_imgs[i].index(max(Weighted_imgs[i], key=sum))
    Maxv.append(Z)
    xMaxv.append(max(Weighted_imgs[i], key=sum))
    #print ('Image No. ',i,' Classfied as part of Class', Z )
    
Sum = [] #Max Sum Weight Classification (Not Used)
xSum = defaultdict(list)
for i in range(0,200):
    for x in range(0,10):
        Y = sum(Weighted_imgs[i][x])
        xSum[i].append(Y)
    Z = xSum[i].index(max(xSum[i]))
    Sum.append(Z)
        #print ('Image No. ',i,' Classfied as part of Class', Z )

MaxT = list(map(int, [Test_Labels][0]))
confusion_matrix(MaxT, Maxv, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.matshow(confusion_matrix(MaxT, Maxv))
plt.show()

#%% Part Three - Validation

Validation_Path='C:\\Users\\khled\\Dropbox\\Masters\\Courses\\xDone\\Machine Learning\\Assigments\\Assignment 1\\Validation'
os.chdir(Validation_Path)
Validation_Labels = np.loadtxt('Validation Labels.txt')
files=os.listdir(Validation_Path)
files.pop()
files = sorted(files,key=lambda x: int(os.path.splitext(x)[0]))

Validation=[] #Importing and Reshaping
for i in files:    
    img=misc.imread(i)
    type(img)
    img.shape
    img=img.reshape(784,)
    img=np.append(img,1)
    Validation.append(img)

etas = [] #Creating New Weights by "Multiplying each weigh with 10 diffrent etas" (10w x 10e)
for x in range(0,len(eta)):
    difeta = []
    for n in range(0,len(W)):
        Y = np.dot(W[n].T, eta[x])
        difeta.append(Y)
    etas.append(difeta)

Max_Weighted_imgs =  [] #Validating each Validation_img with the 10 new Weights (10e x 200i x 10w)
for x in range(0,len(etas)):
    xWeighted_imgs = []
    for n in range(0,len(Validation)):
        xWeighted = []
        for i in range(0,10):
            t = (np.dot(etas[x][i].T,Validation[n]))
            xWeighted.append(t)
        xWeighted_imgs.append(xWeighted)
    Max_Weighted_imgs.append(xWeighted_imgs)

Sortedetas = [] #Sorting Weights (200i x 10e x 10w)
for x in range(0,200):
    xetas = []
    for i in range(0,10):   #01 #0
        xetas.append(Max_Weighted_imgs[i][x])
    Sortedetas.append(xetas)

Maxetas = [] #Getting Best etas by indexing Maximum values
xMaxetas = []
for x in range(0,200):
    Maxw = []
    for i in range(0,10):  
        Z = max(Sortedetas[x][i], key=sum)
        Maxw.append(Z)
    xMaxetas.append(Maxw)
    Maxetas.append(xMaxetas[x].index(max(xMaxetas[x], key=sum)))
    #print ('Image No. ',M,' with Best eta ', i ,' Classfied as part of Class ', xMaxetas[M].index(max(xMaxetas[M], key=sum)))