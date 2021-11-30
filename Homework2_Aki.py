#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:44:09 2021

@author: akihitomaruya
"""
import scipy.io as spio
from trichromacy import human_color_matcher
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import seaborn as sn
import pandas as pd
import numpy.matlib
from mpl_toolkits.mplot3d import Axes3D
ColMatch=spio.loadmat('/Users/akihitomaruya/Desktop/Courses/Class2021/Fall/Math tool for cognitive and neural science/hw2-files/colMatch.mat')
np.random.seed(12)
# a random light
test_light=np.random.rand(31,1)
# primaries 
primaries=ColMatch['P']
# wavelength spectrum at each point
wavelengths=np.arange(400,701,step=10).reshape(-1,1)
#%% Experiment
# Run an experiment and get knob settings 
knobs=human_color_matcher(test_light,primaries)
# derive the combined light that the observer perceived 
perceived_light=primaries@knobs
#%% Visualize
# visualize perceived light and physical light (test_light) 
plt.figure()
test=plt.plot(wavelengths,test_light,'r-',label='Physical light')
p=plt.plot(wavelengths,perceived_light,'b-',label='Perceived light')
plt.xlabel('Wavelengths')
plt.ylabel('Power')
plt.legend()

# They look different but these two lights will generate the same color for human because
# three cones will have the same responses. 

#%% Question b
# Primary from Dr.E's lab

eP=ColMatch['eP']

# Expresssion
# H_e = (HP_e)^(-1)H
# Let's figure out H by feeding impulse responses
impulse=np.identity(len(eP))
H=human_color_matcher(impulse,primaries)
H_e=inv(H@eP)@H

# check if it gives the same knob setting as our lab
knob_E=H_e@test_light

# knob setting should be different since the primary is different 

#%% Question C

Cones=ColMatch['Cones'];
# First, do this informally, by check
#ing that randomly generated lights and their corresponding 3-primary matching lights
#produce equal cone absorptions.
# 10 random lights
test_light=np.random.rand(31,10)
cones_responses=np.round(Cones@test_light,5)
# Run an experiment and get knob settings 
knobs=H@test_light
# derive the combined light that the observer perceived 
perceived_light=primaries@knobs
pri_cones_responses=np.round(Cones@perceived_light,5)

print(np.equal(cones_responses,pri_cones_responses))

# They have the same responses

#%% Make the statement more mathmatical 


# I will write the math 
H_cones=inv(Cones@primaries)@Cones

print(np.equal(np.round(H,5),np.round(H_cones,5)))


#%% Question 2
Data=spio.loadmat('/Users/akihitomaruya/Desktop/Courses/Class2021/Fall/Math tool for cognitive and neural science/hw2-files/regress1.mat')

#%% Make function
def poly_reg(x,y,pol_order,vis=0,x_new=np.empty(1),y_new=np.empty(1)):
    X=np.zeros((len(x),pol_order+1))
    X_new=np.zeros((len(x_new),pol_order+1))
    for pp in range(pol_order+1):
        X[:,[pp]]=x.reshape(-1,1)**pp
        X_new[:,[pp]]=x_new.reshape(-1,1)**pp
    # get svd 
    u, s, vh = np.linalg.svd(X, full_matrices=True)
    # make diagonal matrix for s
    S=np.zeros(X.shape)
    S[:X.shape[1],:X.shape[1]]=np.diag(s)
    psued_S=np.linalg.pinv(S);
    # Compute B0
    B=vh.T@psued_S@u.T@y
    squared_error=np.sum((y-X@B)**2)
    y_hat=X@B;
    y_new_hat=X_new@B;
    pred_error=np.sum((y_new-y_new_hat)**2)
    if vis==1:
        plt.figure()
        plt.scatter(x,y)
        plt.plot(x,y_hat,color='red')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'{pol_order} Order')
        plt.show()
    return y_hat,squared_error,y_new_hat,pred_error
#%% Examine the polynomial regression
max_order=10;
Y_hat=[];Errors=[];
for ii in range(max_order+1):
    y_hat,error,_,_=poly_reg(Data['x'],Data['y'],ii,1) 
    Y_hat.append(y_hat)
    Errors.append(error)

#%% Plot errors
Errors=pd.DataFrame({'Bias':Errors,'Order':np.arange(0,11,step=1)});
sn.barplot(x='Order',y='Bias',data=Errors)

#%% Plot cross varidation score
num_fold=10
Data_dic={'x': Data['x'].reshape(-1),'y': Data['y'].reshape(-1)}
Data_pd=pd.DataFrame(Data_dic)
Pred_error_all=[]
for ii in range(max_order+1):
    pred_error=0
    for ff in range(num_fold):
        ind=np.arange(int(len(Data_pd)/num_fold)*ff,int(len(Data_pd)/num_fold)*(ff+1))
        Train_x=Data_pd['x'].drop(ind).values
        Train_y=Data_pd['y'].drop(ind).values
        Test=Data_pd.iloc[ind]
        y_hat,squared_error,y_new_hat,pred_error=poly_reg(Train_x,Train_y,ii,0,Test['x'].values,Test['y'].values)
        pred_error+=pred_error
    Pred_error_all.append(pred_error)
Errors['Variance']=Pred_error_all
Errors_reshaped=pd.melt(Errors, id_vars='Order', var_name=['Error type'], value_name='Errors')
print(Errors_reshaped)
sn.factorplot(x='Order', y='Errors', hue='Error type', data=Errors_reshaped, kind='bar')
#%% Question 3 
DataQ3=spio.loadmat('/Users/akihitomaruya/Desktop/Courses/Class2021/Fall/Math tool for cognitive and neural science/hw2-files/constrainedLS.mat')

# Visualize data
plt.figure()
plt.scatter(DataQ3['data'][:,0],DataQ3['data'][:,1])
plt.quiver(0,0,DataQ3['w'][0],DataQ3['w'][1])
plt.hlines(0, -max(abs(DataQ3['data'][:,0])), max(abs(DataQ3['data'][:,0])))
plt.vlines(0,-max(abs(DataQ3['data'][:,1])), max(abs(DataQ3['data'][:,1])))


#%% Compute svd

D=DataQ3['data'];w=DataQ3['w'];
u, s, vh = np.linalg.svd(D, full_matrices=True)
S=np.zeros(D.shape)
S[:D.shape[1],:D.shape[1]]=np.diag(s)
S_star_inv=np.linalg.pinv(S)[:2,:2];
w_hat=S_star_inv@vh@w
ganma=1/(w_hat.T@w_hat)
B_opt_hat=ganma*w_hat
transformed_data=D@vh.T@S_star_inv

plt.figure()
plt.scatter(transformed_data[:,0],transformed_data[:,1])
plt.hlines(0, -max(abs(transformed_data[:,0])), max(abs(transformed_data[:,0])))
plt.vlines(0,-max(abs(transformed_data[:,1])), max(abs(transformed_data[:,1])))
plt.quiver(0,0,B_opt_hat[0],B_opt_hat[1],color='r',label='B_hat',angles='xy', scale_units='xy', scale=1)
plt.quiver(0,0,w_hat[0],w_hat[1],color='b',label='W_hat',angles='xy', scale_units='xy', scale=1)
plt.axis([-15, 15, -15, 15]) 
plt.legend()

#%% C
B_opt=vh.T@S_star_inv@B_opt_hat
# solve for total least square (|u|=1)
B_opt2=vh[:,1];


plt.figure()
plt.scatter(DataQ3['data'][:,0],DataQ3['data'][:,1])
plt.hlines(0, -max(abs(DataQ3['data'][:,0])), max(abs(DataQ3['data'][:,0])))
plt.vlines(0,-max(abs(DataQ3['data'][:,1])), max(abs(DataQ3['data'][:,1])))
plt.quiver(0,0,B_opt[0],B_opt[1],color='r',label='B_opt',angles='xy', scale_units='xy', scale=1)
plt.quiver(0,0,w[0],w[1],color='b',label='w',angles='xy', scale_units='xy', scale=1)
plt.quiver(0,0,B_opt2[0],B_opt2[1],color='k',label='B_opt2 (total least square)',angles='xy', scale_units='xy', scale=1)
plt.axis([-2, 2, -2, 2]) 
plt.legend()


#%% 4 Dimensionality reduction with PCA
DataQ4=spio.loadmat('/Users/akihitomaruya/Desktop/Courses/Class2021/Fall/Math tool for cognitive and neural science/hw2-files/windowedSpikes.mat')
DataQ4=DataQ4['data']
plt.figure()
plt.plot(np.arange(0,150,step=1).reshape(-1,1),DataQ4.T)
plt.xlabel('Time (ms)')
plt.ylabel('Action potential')
plt.show()
D=DataQ4;
from sklearn.preprocessing import StandardScaler
D_scaled = StandardScaler().fit_transform( D )

# Covariance matrix 

Corr=1/(400)*D_scaled.T@D_scaled

sn.heatmap(Corr, annot=False, fmt='g')
plt.show()
#(a) Count the number of overlap by finding high cirrelation and subtract from all

#(b)

# Compute svd of the data point
# get svd 
u, s, vh = np.linalg.svd(D_scaled, full_matrices=True)
# Plot the eigen value for first ten
plt.figure()
plt.bar(np.arange(1,11,step=1),s[:10]/sum(s)*100)
plt.xticks(np.arange(1,11,step=1))
plt.xlabel('Ordered factors')
plt.ylabel('Explained variance (%)')

# C Measure the length of the projection of each of the 400 spike waveforms onto the top
#two principal components of the dataset, and plot the resulting values as points in
#2 dimensions. Describe what you see. Can you deduce how many distinct neurons
#produced the 400 voltage traces?
p1=D_scaled@vh[[0],:].T
length_p1=max(p1)-min(p1)
p2=D_scaled@vh[[1],:].T
length_p2=max(p2)-min(p2)
# data is more spread along p1
plt.figure()
plt.scatter(p1,p2)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
# I see four clusters. So, I think there are four neurons 
# (d) Now project each waveform onto the top three principal axes, and plot in 3 dimensions
#(you may want to spin it around, using rotate3d in matlab). Are there any signi cant
#changes you see? Using the 3D plot, can you inform Drs. Bell and Zell how many
#neurons they likely recorded from?

p3=D_scaled@vh[[2],:].T


fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(p1, p2, p3,  edgecolor="k")
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.show()

# There is not much difference. So, they likely recorded from four neurons. 
# Let's use KMean cluster to show it is right


from sklearn.cluster import KMeans
Data_dic={'PC1':p1.reshape(-1),'PC2':p2.reshape(-1)}
Data=pd.DataFrame(Data_dic)
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(Data[['PC1','PC2']])
                for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]


plt.figure(figsize=(12, 8))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.show()






