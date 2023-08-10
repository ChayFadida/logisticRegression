# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 17:09:08 2022

@author: חי פדידה
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 14:05:53 2022

@author: חי פדידה
"""


import math
import pandas as pd
import random

def draw_weights(weights):
    weights=pd.DataFrame(weights,columns=["weights"])
    print(weights)

def generate_data(num_of_data,num_of_val):
    features=[]
    weights=[]
    targets=[]
    for i in range(num_of_data):
        val_i =[]
        for j in range(num_of_val):
            val_i.append(random.random())
        features.append(val_i)
        
    for i in range(num_of_val):
        weights.append((random.random()))
    
    for i in range(num_of_data):
        targets.append(random.choice([0,1]))
        
    return features,weights,targets

def draw_data(feature,targets,num_of_val):
    x_i=[]
    for i in range(num_of_val):
        x_i.append('x'+str(i))
    data=pd.DataFrame(feature,columns=x_i)
    data['targets']= targets
    print(data)
    print("\n")
    
def get_weight_sum(single_feature,weights):
    weight_sum=0
    for i in range(len(weights)):
        weight_sum+=single_feature[i]*weights[i]
    return weight_sum

def sigmoid(w_sum):
    val= 1/(1+math.exp(-w_sum))
    return val

def loss_calc(target,pred):
    val = -(target*math.log10(pred)+(1-target)*(math.log10(1-pred)))
    return val

def Gradient_Decent(feature, weights,target,predection,l_rate):
    new_weights=[]
    for i in range(len(weights)):
        new_single_weight= weights[i]+l_rate*(target-predection)*feature[i]
        new_weights.append(new_single_weight)
        
    return new_weights

def train_data(features,weights,targets,num_of_data,epochs):
    all_average_loss=[]
    for j in range(epochs):
        all_losses_per_epoch=[]
        for i in range(num_of_data):
            weight_sum=get_weight_sum(features[i],weights)
            pred=sigmoid(weight_sum)
            loss=loss_calc(targets[i], pred)
            all_losses_per_epoch.append(loss)
            weights=Gradient_Decent(features[i], weights, targets[i], pred, l_rate)
        average_loss=sum(all_losses_per_epoch)/len(all_losses_per_epoch)
        print("epoch",j+1," the average loss is:",average_loss)
        all_average_loss.append(average_loss)
    
    return all_average_loss,weights
    
def draw_graph(epoch_loss):
    graph=pd.DataFrame(epoch_loss)
    graph.plot(kind="line",grid=True,title="epoch loss graph")
    
def get_data():
    l_rate=eval(input("please enter the learning rate: "))
    num_of_data=int(input("please enter the num of data: "))
    num_of_val=int(input("please enter the num of values for each organ in the data: "))
    return l_rate,num_of_data,num_of_val 

def calc_acurrity(features,weights,num_of_data):
        acc=[]
        for i in range(num_of_data):
            weight_sum=get_weight_sum(features[i],weights)
            pred=sigmoid(weight_sum)    
            if targets[i]==1:
                acc.append(pred)
            else:
                acc.append(1-pred)
        acc=sum(acc)/num_of_data
        print("the accuraty is %.2f percent"%(acc*100))
#----------------Main--------------


l_rate,num_of_data,num_of_val=get_data()
features,weights,targets=generate_data(num_of_data, num_of_val)
draw_data(features, targets, num_of_val)
print("The weights are:\n")
draw_weights(weights)
calc_acurrity(features, weights, num_of_data)
epochs=int(input("please enter number of epochs: "))
epoch_loss,weights=train_data(features, weights, targets, num_of_data, epochs)
draw_graph(epoch_loss)
print("\nThe new weights are:\n")
draw_weights(weights)
calc_acurrity(features, weights, num_of_data)

