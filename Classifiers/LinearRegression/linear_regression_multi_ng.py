# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 11:44:21 2019

@author: Admin1
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 10:41:44 2019

@author: Admin1
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from statistics import stdev
from mpl_toolkits.mplot3d import Axes3D

file = open("E:/machine-learning-ex1/ex1/ex1data2.txt")
X_size = []; X_bedroom = []; Y_price = [];
for f in file:
    X_size.append(float(f.split(",")[0]))
    X_bedroom.append(float(f.split(",")[1]))
    Y_price.append(float(f.split(",")[1].rstrip()))

def normalize(X):
    m = mean(X)
    d = stdev(X)
    X = np.vectorize(lambda x:x-m)(X)
    X = np.vectorize(lambda x:x/d)(X)
    return X

X_size = normalize(X_size)
X_bedroom = normalize(X_bedroom)
Y_price = normalize(Y_price)

#function unknown
a = tf.Variable(0.0)
b = tf.Variable(0.0)
c = tf.Variable(0.0)
y1 = a*X_size + b*X_bedroom +c
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#everytime story
loss = tf.reduce_mean(tf.square(y1-Y_price))
optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

#iterate
train_data=[]
lossx = []
for step in range(100):
    evals = sess.run([optimizer, a, b, c])[1:]
    print(step, evals, sess.run(loss))
    train_data.append(evals)
    lossx.append(sess.run(loss))
    W = sess.run(a)
    W1 = sess.run(b)
    b1 = sess.run(c)
    
f_y = np.vectorize(lambda x:W1*x+b1)(X_bedroom)
f=plt.figure(1) 
plt.plot(X_bedroom,Y_price,"ro")
plt.plot(X_bedroom,f_y)
f.show()

g=plt.figure(2)
plt.plot(lossx)
g.show()

h = plt.figure(3)
ax = h.gca(projection="3d")
ax.plot(X_size, X_bedroom,Y_price,"ro")