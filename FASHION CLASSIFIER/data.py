
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

train_df=pd.read_csv(r'E:/OTHER STUFFS/PYTHON/PROGRAMS/AI/NEURAL NETWORKS/FASHION CLASSIFIER/fashion-mnist_train.csv')
test_df=pd.read_csv(r'E:/OTHER STUFFS/PYTHON/PROGRAMS/AI/NEURAL NETWORKS/FASHION CLASSIFIER/fashion-mnist_test.csv')


train_data= np.array(train_df,dtype="float32")
test_data= np.array(test_df,dtype="float32")

x_train= train_data[:,1:]/255
y_train=train_data[:,0]


x_test= test_data[:,1:]/255
y_test= test_data[:,0]

x_train,x_validation,y_train,y_validation =train_test_split(x_train, y_train, test_size=0.33, random_state=42)
image=x_train[500,:].reshape((28,28))
plt.imshow(image)
plt.show()