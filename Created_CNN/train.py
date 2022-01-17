import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import os
import xlsxwriter

#Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


##### IMPORTANT ########

# change name of Exel book (l.24), and from modelsave (l.312), to make each try traceable!!!!!
# _1 done, _2 done, _3 done


wb=xlsxwriter.Workbook('Results_testLab.xlsx')
ws=wb.add_worksheet()
#f.close()
#f= open("Results_.txt","a+")
row=0
ws.write(row, 0, 'Parameters')
ws.write(row, 1, 'Value')
ws.write(row, 2, 'Epoch')
ws.write(row, 3, 'Training acc')
ws.write(row, 4, 'Validation acc')
ws.write(row, 5, 'Validation loss')


#batch_size has to be multiple of the validation and training data
batch_size = 11

ws.write(1,0, 'batch size')
ws.write(1,1, batch_size)

#Prepare input data
classes = os.listdir('training_data')
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = 128
num_channels = 3
ws.write(2,0, 'validation size')
ws.write(2,1, validation_size)
ws.write(3,0, 'img size')
ws.write(3,1, img_size)
ws.write(4,0, 'num channels')
ws.write(4,1, num_channels)

train_path='C:/Users/Jesus/Desktop/CNN_building/image-classifier/training_data'

# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)


print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

ws.write(5,0, 'Number of files in Training-set')
ws.write(5,1,len(data.train.labels))
ws.write(6,0, 'Number of files in Validation-set')
ws.write(6,1,len(data.valid.labels))


session = tf.Session()

# Placeholders and input:
#All the input images are read in dataset.py file and resized to 128 x 128 x 3 size

x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)



##Network graph params
filter_size_conv1 = 3 
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64
    
fc_layer_size = 128


ws.write(7,0, 'filter size conv1')
ws.write(7,1,filter_size_conv1)
ws.write(8,0, 'filter size conv2')
ws.write(8,1,filter_size_conv2)
ws.write(9,0, 'filter size conv3')
ws.write(9,1,filter_size_conv3)
ws.write(10,0, 'num filters conv1')
ws.write(10,1,num_filters_conv1)
ws.write(11,0, 'num filters conv2')
ws.write(11,1,num_filters_conv2)
ws.write(12,0, 'num filters conv3')
ws.write(12,1,num_filters_conv3)
ws.write(13,0, 'fc layer size')
ws.write(13,1,fc_layer_size)
r=14


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

# We use a RELU as our activation function which simply takes the output of max_pool and applies RELU using tf.nn.relu
# All these operations are done in a single convolution layer.
# Function to define a complete convolutional layer.
def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters,
               r):  
    
    ## Define the weights that will be trained.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## Create biases, also to be trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases

    ## Using max-pooling.  
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')

    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)
    
    ws.write(r,0, 'Arch. layer')
    ws.write(r,1,'Convolutional')

    return layer

#The Output of a convolutional layer is a multi-dimensional Tensor. We want to convert this into a one-dimensional tensor
#This is done in the Flattening layer
    

def create_flatten_layer(layer,r):
    
    # Get shape from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features 
    num_features = layer_shape[1:4].num_elements()

    ##Flatten the layer so we have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])
    
    ws.write(r,0, 'Arch. layer')
    ws.write(r,1,'Flatten')

    return layer


def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True,
             r=r):
    
    #Define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    
    ws.write(r,0, 'Arch. layer')
    ws.write(r,1,'Fully connected layer')

    return layer


layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1,
               r=r)

r=r+1

layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2,
               r=r)

r=r+1


layer_conv3= create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3,
               r=r)
r=r+1
          
layer_flat = create_flatten_layer(layer_conv3,r)

r=r+1

layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True,
                     r=r)
r=r+1

layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False,
                     r=r) 
r=r+1

#Predictions
y_pred = tf.nn.softmax(layer_fc2,name='y_pred')
y_pred_cls = tf.argmax(y_pred, dimension=1)

session.run(tf.global_variables_initializer())

#the cost that will be minimized to reach the optimum value of weights
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

ws.write(r,0, 'Learning rate')
ws.write(r,1,1e-4)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer()) 


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss,ws,row):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    #msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    
   
    
    ws.write(row,2,epoch)
    ws.write(row,3,acc*100)
    ws.write(row,4,val_acc*100)
    ws.write(row,5,val_loss)

    #print(msg.format(epoch + 1, acc, val_acc, val_loss))
    return epoch

total_iterations = 0

saver = tf.train.Saver()
def train(num_iteration,wb,ws,row):
    global total_iterations
    
    
    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))    
            row=row+1
            
            epoch=show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss,ws,row)
            print(int(epoch))
            saver.save(session, 'C:/Users/Jesus/Desktop/CNN_building/image-classifier/keys-wallet-back-LR-test') 

        elif  epoch==10:
            wb.close()

            break


    total_iterations += num_iteration

num_iteration=3000
train(num_iteration,wb,ws,row)
