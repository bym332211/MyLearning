# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def imageprepare():
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    file_name='8_1.png'#导入自己的图片地址
    im = cv2.imread(file_name)

    GrayImage = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    GrayImage = cv2.resize(GrayImage, (28, 28))
    rtn,GrayImage = cv2.threshold(GrayImage, 120,255,cv2.THRESH_BINARY)

    cv2.imwrite('gray.jpg', GrayImage)


    #in terminal 'mogrify -format png *.jpg' convert jpg to png
    im = Image.open('gray.jpg').convert('L')


    #
    im.save("sample.png")
    # plt.imshow(im)
    # plt.show()
    tv = list(im.getdata()) #get pixel values

    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in tv]
    #print(tva)
    return tva

def main():
  result = imageprepare()
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))

  # Define loss and optimizer
  y = tf.matmul(x, W) + b
  y_ = tf.nn.softmax(y)
  sess = tf.Session()
  saver = tf.train.Saver()
  saver.restore(sess, './model.ckpt')
  print("Model restored.")
  result = sess.run(y_, feed_dict={x: [result]})
  print(result)
  code = ""
  for i in result:
      temp = list(i)
      code += chr(temp.index(max(temp)) + 97)
      print(temp.index(max(temp)))


if __name__ == '__main__':
  main()