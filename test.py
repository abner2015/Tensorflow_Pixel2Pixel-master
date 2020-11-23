#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author:abner
@file:test.py
@ datetime:2020/11/20 13:48
@software: PyCharm

"""

from Utils import *
from Pixel2Pixel import *
from Pixel2Pixel_Loss import *
import cv2
from matplotlib import pyplot as plt

from tensorflow.python.tools import inspect_checkpoint as chkp

def predict():
    input_var = tf.placeholder(tf.float32, [None, 256, 256, 3])

    # sess.run(input_var)
    generator = Generator(input_var,reuse = False)
    #必须放在放在模型初始化之后，不然会出现模型参数未初始化的错误
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    checkpoint_path = "./model"
    saver = tf.train.import_meta_graph('./model/Pixel2Pixel_1.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./model'))
    src_path = "D:/abner/project/dataset/house/dataset\mydataset/image/2.jpg"
    src_image = cv2.imread(src_path)
    generator_graph = tf.get_default_graph()
    print(type(generator_graph))


    src_image = cv2.resize(src_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    label = generator_graph.get_tensor_by_name("Generator/Generator_tanh:0")
    input = generator_graph.get_tensor_by_name("Placeholder:0")
    vars_list = tf.train.list_variables(checkpoint_path)
    print(vars_list)
    pred = sess.run(label, feed_dict={input: [src_image]})[0]
    # pred = Generator(input_var)
    # pred = pred.eval(session=sess)
    # plt.imshow(pred)
    # # plt.figure()
    # # plt.imshow(re/255.0)
    # plt.show()
    cv2.imshow("Generator image",pred)
    cv2.waitKey(0)
    # generator = Generator()


    # plt.imshow(gen_output[0, ...])

def p():
    OUTPUT_IMAGE_CHANNEL = 3
    INPUT_IMAGE_CHANNEL = 3
    # 2. build 创建输入数据
    src_input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, OUTPUT_IMAGE_CHANNEL])
    dst_input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_IMAGE_CHANNEL])
    # 创建生成器
    dst_image_op = Generator(src_input_var, reuse=False)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('./model'))

    MAX_EPOCH = 1
    src_path = "D:/abner/project/dataset/house/dataset\mydataset/label/2.png"
    src_image = cv2.imread(src_path)
    src_image = cv2.resize(src_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    images1 = src_image.astype(np.float32) / 127.5 - 1
    pred = sess.run(dst_image_op, feed_dict={src_input_var: [src_image]})[0]
    cv2.imshow("d", pred)
    cv2.waitKey(0)


def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15,15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

def show_checkpoint():
    print(chkp.print_tensors_in_checkpoint_file("./model/Pixel2Pixel_1.ckpt", tensor_name='', all_tensors=True))

if __name__ == '__main__':
    # p()
    predict()
    # show_checkpoint()