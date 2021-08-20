# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#get data
X = np.random.rand(100).astype(np.float32)
Y= 3*X+2
#get output
Y = np.vectorize(lambda y: y+np.random.normal(loc = 0.0, scale = 0.1))(Y)

#function unknown
a = tf.Variable(0.0)
b = tf.Variable(0.0)
y1 = a*X+b
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#everytime story
loss = tf.reduce_mean(tf.square(y1-Y))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

#iterate
train_data=[]
lossx = []
for step in range(1000):
    evals = sess.run([train,a,b])[1:]
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