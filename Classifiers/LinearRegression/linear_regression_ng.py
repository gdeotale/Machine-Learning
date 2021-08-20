# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 10:41:44 2019

@author: Admin1
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

file = open("E:/machine-learning-ex1/ex1/ex1data1.txt")
X = []; Y = [];
for f in file:
    X.append(float(f.split(",")[0]))
    Y.append(float(f.split(",")[1].rstrip()))

#function unknown
a = tf.Variable(0.0)
b = tf.Variable(0.0)
y1 = a*X+b
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#everytime story
loss = tf.reduce_mean(tf.square(y1-Y))
optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

#iterate
train_data=[]
lossx = []
for step in range(5000):
    evals = sess.run([optimizer,a,b])[1:]
    print(step, evals, sess.run(loss))
    train_data.append(evals)
    lossx.append(sess.run(loss))
    W = sess.run(a)
    b1 = sess.run(b)
    
print(W,b1)
f_y = np.vectorize(lambda x:W*x+b1)(X)
f=plt.figure(1) 
plt.plot(X,Y,"ro")
plt.plot(X,f_y)
f.show()

g=plt.figure(2)
plt.plot(lossx)
g.show()