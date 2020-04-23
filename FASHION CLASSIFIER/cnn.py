import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dense,Flatten,Dropout
from keras.optimizers import adam
from keras.callbacks import TensorBoard

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split



# preprocessing of data 
train_df=pd.read_csv(r'E:/OTHER STUFFS/PYTHON/PROGRAMS/AI/NEURAL NETWORKS/FASHION CLASSIFIER/fashion-mnist_train.csv')
test_df=pd.read_csv(r'E:/OTHER STUFFS/PYTHON/PROGRAMS/AI/NEURAL NETWORKS/FASHION CLASSIFIER/fashion-mnist_test.csv')

train_data= np.array(train_df,dtype="float32")
test_data= np.array(test_df,dtype="float32")

x_train= train_data[:,1:]/255
y_train=train_data[:,0]

x_test= test_data[:,1:]/255
y_test= test_data[:,0]

x_train,x_validation,y_train,y_validation =train_test_split(x_train, y_train, test_size=0.33, random_state=42)


# defining the cnn

#model shaping
im_rows=28
im_cols=28
batch_size=512
im_shape=(im_rows,im_cols,1)

x_train=x_train.reshape(x_train.shape[0],*im_shape)
x_test=x_test.reshape(x_test.shape[0],*im_shape)
x_validation=x_validation.reshape(x_validation.shape[0],*im_shape)

#creating the model
cnn_model=Sequential([

    Conv2D(filters=32,kernel_size=3, activation="relu",input_shape=im_shape),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.2),
    Flatten(),
    Dense(32,activation="relu"),
    Dense(10,activation="softmax")
])
#compiling
cnn_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=adam(lr=0.001),
    metrics=["accuracy"]
)

# training the model
cnn_model.fit(

    x_train,y_train,batch_size=batch_size,epochs=3 , 
    validation_data=(x_validation,y_validation) 
)

score=cnn_model.evaluate(x_test,y_test)

print("test loss: {}".format(score[0]))
print("test accuracy: {}".format(score[1]))










