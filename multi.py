import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import math
from astropy.io import fits
import time
import os
import cv2
from scipy.ndimage.filters import gaussian_filter

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow_probability as tfp
tfd = tfp.distributions
#------------------------------------------------------------------

xs = 800
ys = 800


def mask(image, size):
  blure = gaussian_filter(image, size)
  return (image - blure * 0.83)


def show_array(array, name):
    bg = np.percentile(array, 0.1)
    array = array - bg
    array = array / np.percentile(array, 99.8)
    
    cv2.imshow(name, array*0.9)
    cv2.waitKey(1)

#------------------------------------------------------------------

X0 = 60
Y0 = 60
XS = 700
YS = 700

def benoit_noise(image):

  dnoise = 0

  dnoise = dnoise + tf.math.reduce_mean(tf.math.square(image - tf.roll(image, shift=[1,0], axis=[0,1])))
  dnoise = dnoise + tf.math.reduce_mean(tf.math.square(image - tf.roll(image, shift=[0,1], axis=[0,1])))
  dnoise = dnoise + tf.math.reduce_mean(tf.math.square(image - tf.roll(image, shift=[0,-1], axis=[0,1])))
  dnoise = dnoise + tf.math.reduce_mean(tf.math.square(image - tf.roll(image, shift=[-1,0], axis=[0,1])))

  dnoise = dnoise + 0.5*tf.math.reduce_mean(tf.math.square(image - tf.roll(image, shift=[1,1], axis=[0,1])))
  dnoise = dnoise + 0.5*tf.math.reduce_mean(tf.math.square(image - tf.roll(image, shift=[-1,1], axis=[0,1])))
  dnoise = dnoise + 0.5*tf.math.reduce_mean(tf.math.square(image - tf.roll(image, shift=[1,-1], axis=[0,1])))
  dnoise = dnoise + 0.5*tf.math.reduce_mean(tf.math.square(image - tf.roll(image, shift=[-1,-1], axis=[0,1])))

  return tf.math.sqrt(dnoise)


#------------------------------------------------------------------
P = 21
P2 = 10

def get_image(fn):
    x = fits.getdata(fn, ignore_missing_end=True)
    x = x[X0:X0+XS, Y0:Y0+YS]
    x = x.astype(np.float32)

    x = x / np.percentile(x, 90.0)
    x = x - np.percentile(x, 1.0)
    return x


im0 = get_image("./i1.fits")
im1 = get_image("./i3.fits")
im2 = get_image("./i2.fits")
im3 = get_image("./i4.fits")

sum = im0 + im1 + im2 + im3

sum = sum / 4.0

k0 = np.zeros((P, P))*0.0
k0[P2,P2] = 1.0
k0 = gaussian_filter(k0, sigma=2.1) + 0.0001

k1 = np.zeros((P, P))*0.0
k1[P2+2,P2] = 1.0
k1 = gaussian_filter(k1, sigma=2.1) + 0.0001


k2 = np.zeros((P, P))*0.0
k2[P2,P2-2] = 1.0
k2 = gaussian_filter(k2, sigma=2.1) + 0.0001

k3 = np.zeros((P, P))*0.0
k3[P2,P2] = 1.0
k3 = gaussian_filter(k3, sigma=2.1) + 0.0001


k0 = k0 / np.sum(k0)
k0 = k0[:,:]
k1 = k1 / np.sum(k1)
k1 = k1[:,:]
k2 = k2 / np.sum(k2)
k2 = k2[:,:]
k3 = k3 / np.sum(k3)
k3 = k3[:,:]


#------------------------------------------------------------------





psf0 = tf.Variable(k0, name = 'psf0', dtype=tf.float32)
psf1 = tf.Variable(k1, name = 'psf1', dtype=tf.float32)
psf2 = tf.Variable(k2, name = 'psf2', dtype=tf.float32)
psf3 = tf.Variable(k3, name = 'psf3', dtype=tf.float32)

observed0 = tf.constant(im0, name = 'observed0')
observed1 = tf.constant(im1, name = 'observed1')
observed2 = tf.constant(im2, name = 'observed2')
observed3 = tf.constant(im3, name = 'observed3')

mul0 = tf.Variable(1.0, name = 'mul0', dtype=tf.float32)
mul1 = tf.Variable(1.0, name = 'mul1', dtype=tf.float32)
mul2 = tf.Variable(1.0, name = 'mul2', dtype=tf.float32)
mul3 = tf.Variable(1.0, name = 'mul3', dtype=tf.float32)

model = tf.Variable(sum * 1.0, name = 'model', dtype=tf.float32)


#tmp = tf.math.abs(psf0)
#p0 = tmp / tf.reduce_sum(tmp)
p0 = tf.math.abs(psf0)

r0 = mul0 * tf.nn.conv2d(model[tf.newaxis, :, :, tf.newaxis],p0[:, :, tf.newaxis, tf.newaxis],strides=[1, 1, 1, 1],padding="VALID")[0, :, :, 0]

#tmp = tf.math.abs(psf1)
p1 = tf.math.abs(psf1)
#p1 = tmp / tf.reduce_sum(tmp)

r1 = mul1 * tf.nn.conv2d(model[tf.newaxis, :, :, tf.newaxis],p1[:, :, tf.newaxis, tf.newaxis],strides=[1, 1, 1, 1],padding="VALID")[0, :, :, 0]

#tmp = tf.math.abs(psf2)
p2 = tf.math.abs(psf2)

#p2 = tmp / tf.reduce_sum(tmp)

r2 = mul2 * tf.nn.conv2d(model[tf.newaxis, :, :, tf.newaxis],p2[:, :, tf.newaxis, tf.newaxis],strides=[1, 1, 1, 1],padding="VALID")[0, :, :, 0]

#tmp = tf.math.abs(psf3)
p3 = tf.math.abs(psf3)

#p3 = tmp / tf.reduce_sum(tmp)

r3 = mul3 * tf.nn.conv2d(model[tf.newaxis, :, :, tf.newaxis],p3[:, :, tf.newaxis, tf.newaxis],strides=[1, 1, 1, 1],padding="VALID")[0, :, :, 0]


v0 = tf.math.subtract(r0, observed0[P2:XS-P2, P2:YS-P2])
v0 = tf.math.multiply(v0, v0)
e0 = tf.math.reduce_mean(tf.math.square(v0))

v1 = tf.math.subtract(r1, observed1[P2:XS-P2, P2:YS-P2])
v1 = tf.math.multiply(v1, v1)
e1 = tf.math.reduce_mean(tf.math.square(v1))

v2 = tf.math.subtract(r2, observed2[P2:XS-P2, P2:YS-P2])
v2 = tf.math.multiply(v2, v2)
e2 = tf.math.reduce_mean(tf.math.square(v2))

v3 = tf.math.subtract(r3, observed3[P2:XS-P2, P2:YS-P2])
v3 = tf.math.multiply(v3, v3)
e3 = tf.math.reduce_mean(tf.math.square(v3))

noise = benoit_noise(model)

noise = noise - 0.086562345
noise = tf.clip_by_value(noise, 0, 100.0)

loss = 100.0*(e0 + e1 + e2 + e3) + 30.0*noise

#------------------------------------------------------------------

optimizer0 = tf.train.AdamOptimizer(learning_rate=0.0001)
train0 = optimizer0.minimize(loss, var_list=[model, psf0, mul1, mul2, mul3, mul0])

optimizer1 = tf.train.AdamOptimizer(learning_rate=0.0001)
train1 = optimizer1.minimize(loss, var_list=[model, psf1, mul1, mul2, mul3, mul0])

optimizer2 = tf.train.AdamOptimizer(learning_rate=0.0001)
train2 = optimizer0.minimize(loss, var_list=[model, psf2, mul1, mul2, mul3, mul0])

optimizer3 = tf.train.AdamOptimizer(learning_rate=0.0001)
train3 = optimizer1.minimize(loss, var_list=[model, psf3, mul1, mul2, mul3, mul0])

init = tf.initialize_all_variables()

#------------------------------------------------------------------

def optimize():
    np.set_printoptions(linewidth=285, nanstr='nan', precision=2, suppress=True)
    with tf.Session() as session:
        session.run(init)
        mm = 0
        nm = 1.0
        
        for k in range(11100):
            for i in range(600):
                if (i > 0 and i < 150):
                    session.run(train0)
                if (i > 150 and i < 300):
                    session.run(train1)
                if (i > 300 and i < 450):
                    session.run(train2)
                if (i > 450 and i < 600):
                    session.run(train3)
                    
                
                if (i % 150 == 0):
                    print(nm, k, i, "model",
                         session.run(loss),
                         session.run(e0),
                         session.run(e1),
                         session.run(e2),
                         session.run(e3),
                         session.run(mul0),
                         session.run(mul1),
                         session.run(noise))
                         
                         
                    array = session.run(model)
                    a1 = array[:, :]
                    a1 = a1[20:-20,20:-20]
                    a1 = a1 - np.min(a1)
                    mm = np.max(a1)
                    a1 = a1 / np.percentile(a1, 90)
                    show_array(mask(a1,3),"model")
                    cv2.waitKey(1)

                    array = session.run(p0)
                    a1 = array[:,:]
                    a1 = a1 / np.max(a1)
                    
                    cv2.imshow("psf0", a1*0.8)
                    cv2.waitKey(1)

                    array = session.run(p1)
                    a1 = array[:,:]
                    a1 = a1 / np.max(a1)
                    
                    cv2.imshow("psf1", a1*0.8)
                    cv2.waitKey(1)


                    array = session.run(p2)
                    a1 = array[:,:]
                    a1 = a1 / np.max(a1)
                    
                    cv2.imshow("psf2", a1*0.8)
                    cv2.waitKey(1)

                    array = session.run(p3)
                    a1 = array[:,:]
                    a1 = a1 / np.max(a1)
                    
                    cv2.imshow("psf3", a1*0.8)
                    cv2.waitKey(1)


 
 
                    array = session.run(v2)
                    array = array / np.max(array)
                    
                    cv2.imshow("val2", array*1.0)
                    cv2.waitKey(1)



#------------------------------------------------------------------


optimize()

