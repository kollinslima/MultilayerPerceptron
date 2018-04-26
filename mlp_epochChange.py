from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras import optimizers
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import math

number_input = 64
number_classes = 10
hidden_layers = 1
index_layer = 0
neurons_hidden = 60
funct_activation = 'relu'

learning_rate = 0.0001
loss_function = 'categorical_crossentropy'
net_metrics = ['accuracy']
epochs_number = 64

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

train_data = (train_data - train_min)/train_max
validation_data = (validation_data - train_min)/train_max
test_data = (test_data - train_min)/train_max

#####################CREATE MLP############################
mlp = Sequential()

#First Layer and Input
mlp.add(Dense(neurons_hidden, activation=funct_activation, input_dim=number_input))

#All other layers
for index_layer in range(1, hidden_layers):
    mlp.add(Dense(neurons_hidden, activation=funct_activation))

#Output Layer
mlp.add(Dense(number_classes, activation=funct_activation))

net_optimizer = optimizers.RMSprop(lr=learning_rate)

mlp.compile(optimizer=net_optimizer, loss=loss_function, metrics=net_metrics)

history = mlp.fit(train_data, train_label_one_hot,epochs=epochs_number, verbose=1,validation_data=(validation_data, validation_label_one_hot))

max_value = 0
max_epoch = 0

for i in range(0,len(history.history['val_loss'])):
    if math.isnan(history.history['val_acc'][i]) or math.isnan(history.history['val_loss'][i]):
        continue
    if history.history['val_acc'][i] > max_value:
        max_value = history.history['val_acc'][i]
        max_epoch = i+1

print("Max Epoch: {}".format(max_epoch))
plt.figure(figsize=[8,6])
plt.xlim(xmax=64)
plt.plot(history.history['loss'], 'r')
plt.plot(history.history['val_loss'], 'b')
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.savefig('Loss_epochs.png')
plt.close()

plt.figure(figsize=[8,6])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['Training accuracy', 'Validation accuracy'],fontsize=18)
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.savefig('Accuracy_epochs.png')
plt.close()
