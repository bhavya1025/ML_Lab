import pandas as pd
import numpy as np


data= pd.read_csv('data.csv')
attributes= np.array(data)[:,:-1]
target= np.array(data)[:,-1]


print(data)

for i,value in enumerate(target):   #to get first hypothesis
  if value=="yes":
    hyp= attributes[i].copy()
    print("The specific hypothesis is ", hyp)
    break


for i, value in enumerate(attributes):
  if target[i]=="yes":
    for x in range(len(hyp)):
      if value[x]!= hyp[x]:
        hyp[x]='?'
        print(hyp)
      else:
        pass


print("hypothesis is ", hyp)
