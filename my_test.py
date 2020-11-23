import glob
import time

from Pixel2Pixel import *
from Pixel2Pixel_Loss import *
from Utils import *


OUTPUT_IMAGE_CHANNEL = 3
INPUT_IMAGE_CHANNEL = 3
# 2. build 创建输入数据
src_input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, OUTPUT_IMAGE_CHANNEL])
dst_input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_IMAGE_CHANNEL])
#创建生成器
dst_image_op = Generator(src_input_var, reuse = False)

#创建判别器
# D_fake_op = Discriminator(dst_image_op, src_input_var, reuse = False)
# D_real_op = Discriminator(dst_input_var, src_input_var, reuse = True)

# vars = tf.trainable_variables()





# 6. train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('./model'))



MAX_EPOCH = 1
src_path = "D:/abner/project/dataset/house/dataset\mydataset/image/31830270.jpg"
src_image = cv2.imread(src_path)
src_image = cv2.resize(src_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
images1 = src_image.astype(np.float32) / 127.5 - 1
pred = sess.run(dst_image_op, feed_dict={src_input_var: [src_image]})[0]
cv2.imshow("d", pred)
cv2.waitKey(0)
