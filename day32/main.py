import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#Mmnist data - Bunch of training data with (28,28) size pixels with respective labels.
mnist = tf.keras.datasets.mnist

(x_train,y_train) , (x_test,y_test) = mnist.load_data()#Loading and storing the values in the two tuples,training and testing data

#Preprocessing - Normalizing
x_train = tf.keras.utils.normalize(x_train,axis=1)#Scaling the data/Normalizing 
x_test = tf.keras.utils.normalize(x_test,axis=1)

""" model = tf.keras.models.Sequential()#Basic sequential Neural Network
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))#Flattens the 28,28 pixels into a big flat 784 pixels instead of a grid like structure
model.add(tf.keras.layers.Dense(128,activation ='relu'))#Each Neuron/Units is connected to each other Neuron of other layers || 128 neurons and the non-linear activation function - ReLu ( Rectified Linear Unit)
model.add(tf.keras.layers.Dense(1288,activation ='relu'))#Second Layer
model.add(tf.keras.layers.Dense(10,activation ='softmax'))#Output  layer || Softmax - Calculates the probability for every class and outputs them. The sum of probability of all the calculates equates to 1

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])#Compile the model

model.fit(x_train,y_train,epochs=20) #Epochs - How many iterations is it going to run for ?

model.save('handwriting.model') """ 

model = tf.keras.models.load_model('handwriting.model') #Loading the model
 



 
for n in range(0,10):
    if os.path.isfile(f'digits/{n}_digit.png'):
        img = cv2.imread(f'digits/{n}_digit.png')[:,:,0]#We are not interested in colors,we are only interested in the shape/form of the number,therefore we only get the first channel
        img = np.invert(np.array([img])) #It's white on black and not black on white, therefore we invert it and make it into a numpy array
        prediction = model.predict(img)
        print(f"The digit is probably a :{np.argmax(prediction)} ")#np.argmax - Outputs the class with the highest probability
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show() 
    else:
        print("Error for input : %s" %n)

print("Successfully done for digits set starting with '_'\n Processing for different datasets starting from digit0 to digit9")

for n in range(0,10):
    if os.path.isfile(f'digits/digit{n}.png'):
        img = cv2.imread(f'digits/digit{n}.png')[:,:,0]#We are not interested in colors,we are only interested in the shape/form of the number,therefore we only get the first channel
        img = np.invert(np.array([img])) #It's white on black and not black on white, therefore we invert it and make it into a numpy array
        prediction = model.predict(img)
        print(f"The digit is probably a :{np.argmax(prediction)} ")#np.argmax - Outputs the class with the highest probability
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show() 
    else:
        print("Error for input : %s" %n)
print("Done!")

"""
OBSERVATIONS : 

for _digit - 5/10 is correct - epoch - 3

For epoch - 10
	for _digit - 4/10 correct
	
for epoch-10 ,hidden layer neurons - 256, output neurons - 20
	_digit = 3/10 (worse)

for epochs = 20, hidden layer neurons - 128, output layer neurons = 10
	_digit = 5/10
	digit(0:10) - 1/10 ( Bad Performance )


"""





