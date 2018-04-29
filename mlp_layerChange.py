from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras import optimizers
from keras import initializers
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import math

number_input = 64
number_classes = 10
hidden_layers = 1
index_layer = 0
neurons_hidden = 53
funct_activation = 'relu'
funct_activation_output = 'softmax'

initializer_kernel=initializers.random_uniform()
initializer_bias='ones'

learning_rate = 0.005
loss_function = 'categorical_crossentropy'
net_metrics = ['accuracy']
epochs_number = 4

validation_split = 3
##################READ DATABASE - TRAIN#####################
train_read = genfromtxt('dataset/train.csv', delimiter=',')
tmp_train_data = np.array([])
tmp_train_label = np.array([])

for i in range(0, train_read.shape[0]):
    last = train_read[i][-1]
    tmp_train_data = np.append(tmp_train_data, train_read[i][:-1], axis=0)
    tmp_train_label = np.append(tmp_train_label,last)

tmp_train_data = tmp_train_data.reshape(-1, 64)
tmp_train_label = tmp_train_label.reshape(-1,1)

#print(tmp_train_data.shape[0])
#print(tmp_train_label.shape[0])

split_train_validation_data = np.split(tmp_train_data, validation_split)
split_train_validation_label = np.split(tmp_train_label, validation_split)

train_data = np.concatenate((split_train_validation_data[0], split_train_validation_data[1]))
train_label = np.concatenate((split_train_validation_label[0], split_train_validation_label[1]))

validation_data = split_train_validation_data[2]
validation_label = split_train_validation_label[2]

#print(train_data)
#print(train_label)
#print(train_data.shape[0])
#print(train_label.shape[0])
#print(validation_data)
#print(validation_label)
#print(validation_data.shape[0])
#print(validation_label.shape[0])

train_label_one_hot = to_categorical(train_label)
validation_label_one_hot = to_categorical(validation_label)

##################READ DATABASE - TEST#####################
test_read = genfromtxt('dataset/test.csv', delimiter=',')
test_data = np.array([])
test_label = np.array([])

for i in range(0, test_read.shape[0]):
    last = test_read[i][-1]
    test_data = np.append(test_data, test_read[i][:-1], axis=0)
    test_label = np.append(test_label,last)

test_data = test_data.reshape(-1, 64)
test_label = test_label.reshape(-1,1)
test_label_one_hot = to_categorical(test_label)

####################NORMALIZATION########################
train_min = np.amin(train_data)
train_max = np.amax(train_data)

'''
Normalization between a and b
x = (b - a)(x - min)/(max - min) + a
'''
train_data = (2*((train_data - train_min)/(train_max - train_min))) - 1
validation_data = (2*((validation_data - train_min)/(train_max - train_min))) - 1
test_data = (2*((test_data - train_min)/(train_max - train_min))) - 1

#####################CREATE MLP############################
loss_test = []
loss_validation = []
acc_test = []
acc_validation = []

max_value = 0
max_layer = 0
for hidden_layers in range(1,64):
    mlp = Sequential()

    #First Layer and Input
    mlp.add(Dense(neurons_hidden, 
        kernel_initializer=initializer_kernel, 
        bias_initializer=initializer_bias,
        activation=funct_activation, 
        input_dim=number_input))

    #All other layers
    for index_layer in range(1, hidden_layers):
        mlp.add(Dense(neurons_hidden, 
            kernel_initializer=initializer_kernel, 
            bias_initializer=initializer_bias,
            activation=funct_activation))

    #Output Layer
    mlp.add(Dense(number_classes,
        kernel_initializer=initializer_kernel,
        bias_initializer=initializer_bias,
        activation=funct_activation_output))
    
    net_optimizer = optimizers.RMSprop(lr=learning_rate)

    mlp.compile(optimizer=net_optimizer, loss=loss_function, metrics=net_metrics)

    history = mlp.fit(train_data, train_label_one_hot,epochs=epochs_number, verbose=1,validation_data=(validation_data, validation_label_one_hot))

    loss_test.append(history.history['loss'][-1])
    loss_validation.append(history.history['val_loss'][-1])
    acc_test.append(history.history['acc'][-1])
    acc_validation.append(history.history['val_acc'][-1])

    if math.isnan(acc_validation[-1]) or math.isnan(loss_validation[-1]):
        continue

    if acc_validation[-1] > max_value:
        max_value = acc_validation[-1]
        max_layer = hidden_layers

print("Max Layer: {}".format(max_layer))
plt.figure(figsize=[8,6])
plt.xlim(xmax=64)
plt.plot(loss_test, 'r')
plt.plot(loss_validation, 'b')
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Number of Layers',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.savefig('Loss_multiLayer.png')
plt.close()

plt.figure(figsize=[8,6])
plt.plot(acc_test)
plt.plot(acc_validation)
plt.legend(['Training accuracy', 'Validation accuracy'],fontsize=18)
plt.xlabel('Number of Layers',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.savefig('Accuracy_multiLayer.png')
plt.close()
