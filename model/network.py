from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,time,cv2,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc as sic
import subprocess
import numpy as np




def lrelu(x):
    return tf.maximum(x*0.2,x)

def bilinear_up_and_concat(x1, x2, output_channels, in_channels, scope):
    with tf.variable_scope(scope):
        upconv = tf.image.resize_images(x1, [tf.shape(x1)[1]*2, tf.shape(x1)[2]*2])
        upconv.set_shape([None, None, None, in_channels])
        upconv = slim.conv2d(upconv,output_channels,[3,3], rate=1, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),scope='up_conv1')
        upconv_output =  tf.concat([upconv, x2], axis=3)
        upconv_output.set_shape([None, None, None, output_channels*2])
    return upconv_output


# def conv2upconcat(conv_a, conv_b, output_channel, ext):
#     up6 =  bilinear_up_and_concat( conv_b, conv_a, output_channel, output_channel*2, scope=ext+"g_up_1" )
#     conv6=slim.conv2d(up6,  output_channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_1')
#     conv6=slim.conv2d(conv6,output_channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_2')
#     return conv6



def bilinear_resize_and_concat(x1, x2, output_channel, scope):
    with tf.variable_scope(scope) as variable_scope:
        upconv = tf.image.resize_images(x1, [tf.shape(x1)[1] * 2, tf.shape(x1)[2] * 2])
        upconv = slim.conv2d(upconv, output_channel, [3, 3], rate=1, activation_fn=None,
                             weights_initializer=ini, scope="upconv1")
        upconv = tf.concat([upconv, x2], axis=3)
    return upconv


def conv2upconcat(x1, x2, output_channel, scope):
    up1 = bilinear_resize_and_concat(x1, x2, output_channel=output_channel, scope=scope + '_up')
    conv6 = slim.conv2d(up1, output_channel, kernel_size=3, activation_fn=tf.nn.relu,
                        weights_initializer=ini, scope=scope + '_1')
    conv6 = slim.conv2d(conv6, output_channel, kernel_size=3, activation_fn=tf.nn.relu,
                        weights_initializer=ini, scope=scope + '_2')
    return conv6


def conv2pool(input, channel, ext):
    conv1 = slim.conv2d(input, channel, kernel_size=3, rate=1, activation_fn=tf.nn.relu,
                        weights_initializer=ini, scope=ext + '_1')
    conv1 = slim.conv2d(conv1, channel, kernel_size=3, rate=1, activation_fn=tf.nn.relu,
                        weights_initializer=ini, scope=ext + '_2')
    max_pool1 = slim.max_pool2d(conv1, [2, 2], stride=2, padding='SAME')
    return conv1, max_pool1


ini = tf.initializers.he_normal()

def UNet_2decoders(input, channel=32, output_channel=3, directions=4, reuse=False, pred_light=False, ext=""):
    '''
    input should be devidable by 16
    :param input:
    :param channel:
    :param output_channel:
    :param directions: number of light directios
    :param reuse:
    :param pred_light:
    :param ext:
    :return:
    '''
    if reuse:
        tf.get_variable_scope().reuse_variables()
    # input size is (1, 1024, 1216, 90) (1, 256, 256, 10)
    conv1 = slim.conv2d(input, channel, [1, 1], rate=1, activation_fn=tf.nn.relu,
                        weights_initializer=ini, scope=ext + 'g_conv1_1')
    conv1 = slim.conv2d(conv1, channel, [3, 3], rate=1, activation_fn=tf.nn.relu,
                        weights_initializer=ini, scope=ext + 'g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], stride=2, padding='SAME')  # (1, 512, 608, 32) (1, 128, 128, 32)
    conv2, pool2 = conv2pool(pool1, channel * 2, ext=ext + 'g_conv2')  # (1, 256, 304, 64) (1, 64, 64, 64)
    conv3, pool3 = conv2pool(pool2, channel * 4, ext=ext + 'g_conv3')  # (1, 128, 152, 128) (1, 32, 32, 128)
    conv4, pool4 = conv2pool(pool3, channel * 8, ext=ext + 'g_conv4')  # (1, 64, 76, 256) (1, 16, 16, 256)
    conv5 = slim.conv2d(pool4, channel * 16, [3, 3], rate=1, activation_fn=tf.nn.relu,
                        weights_initializer=ini, scope=ext + 'g_conv5_1')
    conv5 = slim.conv2d(conv5, channel * 16, [3, 3], rate=1, activation_fn=tf.nn.relu,
                        weights_initializer=ini, scope=ext + 'g_conv5_2')  # (1, 64, 76, 512) (1, 16, 16, 512)
    with tf.variable_scope('lights'):
        light_ext = "light_"
        BRDF_conv6 = conv2upconcat(conv5, conv4, output_channel=channel * 8, scope=light_ext+'g_conv6')
        # (1, 128, 152, 256)
        BRDF_conv7 = conv2upconcat(BRDF_conv6, conv3, output_channel=channel * 4, scope=light_ext+'g_conv7')  # (1, 256, 304, 128)
        BRDF_conv8 = conv2upconcat(BRDF_conv7, conv2, output_channel=channel * 2, scope=light_ext+'g_conv8')  # (1, 512, 608, 64)
        BRDF_conv9 = conv2upconcat(BRDF_conv8, conv1, output_channel=channel, scope=light_ext+'g_conv9')  # (1, 1024, 1216, 32)
        pred_reflect = slim.conv2d(BRDF_conv9, 3, kernel_size=3, activation_fn=None,
                            weights_initializer=ini, scope=light_ext+'g_conv9_3')

    with tf.variable_scope('normal'):
        conv6 = conv2upconcat(conv5, conv4, output_channel=channel * 8, scope='g_conv6')
        # (1, 128, 152, 256)
        conv7 = conv2upconcat(conv6, conv3, output_channel=channel * 4, scope='g_conv7')  # (1, 256, 304, 128)
        conv8 = conv2upconcat(conv7, conv2, output_channel=channel * 2, scope='g_conv8')  # (1, 512, 608, 64)
        conv9 = conv2upconcat(conv8, conv1, output_channel=channel, scope='g_conv9')  # (1, 1024, 1216, 32)
        pred_trans = slim.conv2d(conv9, 3, kernel_size=3, activation_fn=None,
                            weights_initializer=ini, scope='g_conv9_3')

    return pred_trans, pred_reflect

# def UNet_2decoders(input, channel=32, output_channel=3,reuse=False,ext=""):
#     if reuse:
#         tf.get_variable_scope().reuse_variables()
#     conv1=slim.conv2d(input,channel,[1,1], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_1')
#     conv1=slim.conv2d(conv1,channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_2')
#     pool1=slim.max_pool2d(conv1, [2, 2], padding='same' )
#     conv2=slim.conv2d(pool1,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_1')
#     conv2=slim.conv2d(conv2,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_2')
#     pool2=slim.max_pool2d(conv2, [2, 2], padding='same' )
#     conv3=slim.conv2d(pool2,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv3_1')
#     conv3=slim.conv2d(conv3,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv3_2')
#     pool3=slim.max_pool2d(conv3, [2, 2], padding='same' )
#     conv4=slim.conv2d(pool3,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv4_1')
#     conv4=slim.conv2d(conv4,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv4_2')
#     pool4=slim.max_pool2d(conv4, [2, 2], padding='same' )
#     conv5=slim.conv2d(pool4,channel*16,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv5_1')
#     conv5=slim.conv2d(conv5,channel*16,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv5_2')

#     with tf.variable_scope("Reflection"):
#         r_conv6=conv2upconcat(conv4, conv5, channel*8, "Reflection")
#         r_conv7=conv2upconcat(conv3, r_conv6, channel*4, "Reflection")
#         r_conv8=conv2upconcat(conv2, r_conv7, channel*2, "Reflection")
#         r_up9 =  bilinear_up_and_concat(r_conv8, conv1, channel, channel*2, scope="Reflection"+"g_up_4" )
#         r_conv9=slim.conv2d(r_up9,  channel,[3,3], rate=1, activation_fn=lrelu, scope="Reflection"+'g_conv9_1')
#         pred_reflect=slim.conv2d(r_conv9, output_channel,[3,3], rate=1, activation_fn=None,  scope="Reflection"+'g_conv9_2')

#     with tf.variable_scope("Transmission"):
#         conv6=conv2upconcat(conv4, conv5, channel*8, "Transmission")
#         conv7=conv2upconcat(conv3, conv6, channel*4, "Transmission")
#         conv8=conv2upconcat(conv2, conv7, channel*2, "Transmission")
#         up9 =  bilinear_up_and_concat( conv8, conv1, channel, channel*2, scope="Transmission"+"g_up_4" )
#         conv9=slim.conv2d(up9,  channel,[3,3], rate=1, activation_fn=lrelu, scope="Transmission"+'g_conv9_1')
#         pred_trans=slim.conv2d(conv9, output_channel,[3,3], rate=1, activation_fn=None,  scope="Transmission"+'g_conv9_2')

#     return pred_trans, pred_reflect

def UNet(input, channel=32, output_channel=3,reuse=False,ext=""):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    conv1=slim.conv2d(input,channel,[1,1], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_1')
    conv1=slim.conv2d(conv1,channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='same' )
    conv2=slim.conv2d(pool1,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_1')
    conv2=slim.conv2d(conv2,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='same' )
    conv3=slim.conv2d(pool2,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv3_1')
    conv3=slim.conv2d(conv3,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='same' )
    conv4=slim.conv2d(pool3,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv4_1')
    conv4=slim.conv2d(conv4,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='same' )
    conv5=slim.conv2d(pool4,channel*16,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv5_1')
    conv5=slim.conv2d(conv5,channel*16,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv5_2')

    up6 =  bilinear_up_and_concat( conv5, conv4, channel*8, channel*16, scope=ext+"g_up_1" )
    conv6=slim.conv2d(up6,  channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_1')
    conv6=slim.conv2d(conv6,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_2')
    up7 =  bilinear_up_and_concat( conv6, conv3, channel*4, channel*8, scope=ext+"g_up_2" )
    conv7=slim.conv2d(up7,  channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv7_1')
    conv7=slim.conv2d(conv7,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv7_2')
    up8 =  bilinear_up_and_concat( conv7, conv2, channel*2, channel*4, scope=ext+"g_up_3" )
    conv8=slim.conv2d(up8,  channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv8_1')
    conv8=slim.conv2d(conv8,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv8_2')
    up9 =  bilinear_up_and_concat( conv8, conv1, channel, channel*2, scope=ext+"g_up_4" )
    conv9=slim.conv2d(up9,  channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv9_1')
    conv9=slim.conv2d(conv9, output_channel,[3,3], rate=1, activation_fn=None,  scope=ext+'g_conv9_2')

    return conv9


def UNet_SE(input, channel=32, output_channel=3,reuse=False,ext=""):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    conv1=slim.conv2d(input,channel,[1,1], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_1')
    conv1=slim.conv2d(conv1,channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='same' )
    conv2=slim.conv2d(pool1,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_1')
    conv2=slim.conv2d(conv2,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='same' )
    conv3=slim.conv2d(pool2,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv3_1')
    conv3=slim.conv2d(conv3,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='same' )
    conv4=slim.conv2d(pool3,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv4_1')
    conv4=slim.conv2d(conv4,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='same' )
    conv5=slim.conv2d(pool4,channel*16,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv5_1')
    conv5=slim.conv2d(conv5,channel*16,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv5_2')

    conv5=slim.conv2d(conv5, channel*16,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv5_3')
    global_pooling = tf.reduce_mean(conv5, axis=[0,1,2], keep_dims=True)
    se = slim.fully_connected(global_pooling, channel, activation_fn=tf.nn.relu, scope = ext+'g_fc1')
    ex = slim.fully_connected(global_pooling, channel * 16, activation_fn=tf.nn.relu, scope = ext+'g_fc2')
    attention_channel = tf.nn.sigmoid(ex)
    conv5 = conv5 * attention_channel

    up6 =  bilinear_up_and_concat( conv5, conv4, channel*8, channel*16, scope=ext+"g_up_1" )
    conv6=slim.conv2d(up6,  channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_1')
    conv6=slim.conv2d(conv6,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_2')
    up7 =  bilinear_up_and_concat( conv6, conv3, channel*4, channel*8, scope=ext+"g_up_2" )
    conv7=slim.conv2d(up7,  channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv7_1')
    conv7=slim.conv2d(conv7,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv7_2')
    up8 =  bilinear_up_and_concat( conv7, conv2, channel*2, channel*4, scope=ext+"g_up_3" )
    conv8=slim.conv2d(up8,  channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv8_1')
    conv8=slim.conv2d(conv8,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv8_2')
    up9 =  bilinear_up_and_concat( conv8, conv1, channel, channel*2, scope=ext+"g_up_4" )
    conv9=slim.conv2d(up9,  channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv9_1')
    conv9=slim.conv2d(conv9, output_channel,[3,3], rate=1, activation_fn=None,  scope=ext+'g_conv9_2')

    return conv9



def UNet_global(input, channel=32, output_channel=3,reuse=False,ext=""):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    conv1=slim.conv2d(input,channel,[1,1], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_1')
    conv1=slim.conv2d(conv1,channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='same' )
    conv2=slim.conv2d(pool1,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_1')
    conv2=slim.conv2d(conv2,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='same' )
    conv3=slim.conv2d(pool2,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv3_1')
    conv3=slim.conv2d(conv3,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='same' )
    conv4=slim.conv2d(pool3,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv4_1')
    conv4=slim.conv2d(conv4,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='same' )
    conv5=slim.conv2d(pool4,channel*16,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv5_1')
    conv5=slim.conv2d(conv5,channel*16,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv5_2')

    global_conv1 = slim.conv2d(input,channel,[1,1], rate=1, activation_fn=lrelu, scope=ext+'gl_conv1')
    global_pool1=slim.max_pool2d(global_conv1, [2, 2], padding='same' )
    global_conv2 = slim.conv2d(global_pool1,channel * 2,[1,1], rate=1, activation_fn=lrelu, scope=ext+'gl_conv2')
    global_pool2=slim.max_pool2d(global_conv2, [2, 2], padding='same' )
    global_conv3 = slim.conv2d(global_pool2,channel * 4,[1,1], rate=1, activation_fn=lrelu, scope=ext+'gl_conv3')
    global_pool3=slim.max_pool2d(global_conv3, [2, 2], padding='same' )
    global_conv4 = slim.conv2d(global_pool3,channel * 8,[1,1], rate=1, activation_fn=lrelu, scope=ext+'gl_conv4')
    global_pool4=slim.max_pool2d(global_conv4, [2, 2], padding='same' )
    global_conv5 = slim.conv2d(global_pool4,channel * 16,[1,1], rate=1, activation_fn=lrelu, scope=ext+'gl_conv5')
    global_pooling = tf.reduce_mean(global_conv5, axis=[0,1,2], keep_dims=True)
    se = slim.fully_connected(global_pooling, channel, activation_fn=tf.nn.relu, scope = ext+'g_fc1')
    ex = slim.fully_connected(global_pooling, channel * 16, activation_fn=tf.nn.relu, scope = ext+'g_fc2')
    attention_channel = tf.nn.sigmoid(ex)
    conv5 = conv5 * attention_channel

    up6 =  bilinear_up_and_concat( conv5, conv4, channel*8, channel*16, scope=ext+"g_up_1" )
    conv6=slim.conv2d(up6,  channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_1')
    conv6=slim.conv2d(conv6,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_2')
    up7 =  bilinear_up_and_concat( conv6, conv3, channel*4, channel*8, scope=ext+"g_up_2" )
    conv7=slim.conv2d(up7,  channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv7_1')
    conv7=slim.conv2d(conv7,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv7_2')
    up8 =  bilinear_up_and_concat( conv7, conv2, channel*2, channel*4, scope=ext+"g_up_3" )
    conv8=slim.conv2d(up8,  channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv8_1')
    conv8=slim.conv2d(conv8,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv8_2')
    up9 =  bilinear_up_and_concat( conv8, conv1, channel, channel*2, scope=ext+"g_up_4" )
    conv9=slim.conv2d(up9,  channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv9_1')
    conv9=slim.conv2d(conv9, output_channel,[3,3], rate=1, activation_fn=None,  scope=ext+'g_conv9_2')

    return conv9



def DeepUNet(input, channel=32, output_channel=10,reuse=False,ext=""):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    conv1=slim.conv2d(input,channel,[1,1], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_1')
    conv1=slim.conv2d(conv1,channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='same' )
    conv2=slim.conv2d(pool1,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_1')
    conv2=slim.conv2d(conv2,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='same' )
    conv3=slim.conv2d(pool2,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv3_1')
    conv3=slim.conv2d(conv3,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='same' )
    conv4=slim.conv2d(pool3,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv4_1')
    conv4=slim.conv2d(conv4,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='same' )
    conv5=slim.conv2d(pool4,channel*16,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv5_1')
    conv5=slim.conv2d(conv5,channel*16,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv5_2')
    pool5=slim.max_pool2d(conv5, [2, 2], padding='same' )
    conv6=slim.conv2d(pool6,channel*32,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_1')
    conv6=slim.conv2d(conv7,channel*32,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_2')

    up6 =  bilinear_up_and_concat( conv5, conv4, channel*8, channel*16, scope=ext+"g_up_1" )
    conv6=slim.conv2d(up6,  channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_1')
    conv6=slim.conv2d(conv6,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_2')
    up7 =  bilinear_up_and_concat( conv6, conv3, channel*4, channel*8, scope=ext+"g_up_2" )
    conv7=slim.conv2d(up7,  channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv7_1')
    conv7=slim.conv2d(conv7,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv7_2')
    up8 =  bilinear_up_and_concat( conv7, conv2, channel*2, channel*4, scope=ext+"g_up_3" )
    conv8=slim.conv2d(up8,  channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv8_1')
    conv8=slim.conv2d(conv8,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv8_2')
    up9 =  bilinear_up_and_concat( conv8, conv1, channel, channel*2, scope=ext+"g_up_4" )
    conv9=slim.conv2d(up9,  channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv9_1')
    conv9=slim.conv2d(conv9, 5,[3,3], rate=1, activation_fn=None,  scope=ext+'g_conv9_2')

    return conv9


def R_net_seperate(input, channel=32, output_channel=10,reuse=False,ext=""):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    conv1=slim.conv2d(input,channel,[1,1], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_1')
    conv1=slim.conv2d(conv1,channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )
    conv2=slim.conv2d(pool1,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_1')
    conv2=slim.conv2d(conv2,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )
    conv3=slim.conv2d(pool2,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv3_1')
    conv3=slim.conv2d(conv3,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )
    conv4=slim.conv2d(pool3,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv4_1')
    conv4=slim.conv2d(conv4,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )
    conv5=slim.conv2d(pool4,channel*16,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv5_1')
    conv5=slim.conv2d(conv5,channel*16,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv5_2')


    up6 =  bilinear_up_and_concat( conv5, conv4, channel*8, channel*16, scope=ext+"g_up_1" )
    conv6=slim.conv2d(up6,  channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_1')
    conv6=slim.conv2d(conv6,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_2')
    up7 =  bilinear_up_and_concat( conv6, conv3, channel*4, channel*8, scope=ext+"g_up_2" )
    conv7=slim.conv2d(up7,  channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv7_1')
    conv7=slim.conv2d(conv7,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv7_2')
    up8 =  bilinear_up_and_concat( conv7, conv2, channel*2, channel*4, scope=ext+"g_up_3" )
    conv8=slim.conv2d(up8,  channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv8_1')
    conv8=slim.conv2d(conv8,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv8_2')
    up9 =  bilinear_up_and_concat( conv8, conv1, channel, channel*2, scope=ext+"g_up_4" )
    conv9=slim.conv2d(up9,  channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv9_1')
    conv9=slim.conv2d(conv9, 5,[3,3], rate=1, activation_fn=None,  scope=ext+'g_conv9_2')

    upR6 =  bilinear_up_and_concat( conv5, conv4, channel*8, channel*16, scope=ext+"g_upR_1" )
    convR6=slim.conv2d(upR6,  channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convR6_1')
    convR6=slim.conv2d(convR6,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convR6_2')
    upR7 =  bilinear_up_and_concat( convR6, conv3, channel*4, channel*8, scope=ext+"g_upR_2" )
    convR7=slim.conv2d(upR7,  channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convR7_1')
    convR7=slim.conv2d(convR7,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convR7_2')
    upR8 =  bilinear_up_and_concat( convR7, conv2, channel*2, channel*4, scope=ext+"g_upR_3" )
    convR8=slim.conv2d(upR8,  channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convR8_1')
    convR8=slim.conv2d(convR8,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convR8_2')
    upR9 =  bilinear_up_and_concat( convR8, conv1, channel, channel*2, scope=ext+"g_upR_4" )
    convR9=slim.conv2d(upR9,  channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convR9_1')
    convR9=slim.conv2d(convR9, 5,[3,3], rate=1, activation_fn=None,  scope=ext+'g_convR9_2')

    return tf.concat([conv9,convR9],axis=3)



def net_seperate(input, channel=32, output_channel=10,reuse=False,ext=""):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    conv1=slim.conv2d(input,channel,[1,1], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_1')
    conv1=slim.conv2d(conv1,channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='same' )
    conv2=slim.conv2d(pool1,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_1')
    conv2=slim.conv2d(conv2,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='same' )
    conv3=slim.conv2d(pool2,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv3_1')
    conv3=slim.conv2d(conv3,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='same' )
    conv4=slim.conv2d(pool3,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv4_1')
    conv4=slim.conv2d(conv4,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='same' )
    conv5=slim.conv2d(pool4,channel*16,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv5_1')
    conv5=slim.conv2d(conv5,channel*16,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv5_2')


    up6 =  bilinear_up_and_concat( conv5, conv4, channel*8, channel*16, scope=ext+"g_up_1" )
    conv6=slim.conv2d(up6,  channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_1')
    conv6=slim.conv2d(conv6,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_2')
    up7 =  bilinear_up_and_concat( conv6, conv3, channel*4, channel*8, scope=ext+"g_up_2" )
    conv7=slim.conv2d(up7,  channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv7_1')
    conv7=slim.conv2d(conv7,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv7_2')
    up8 =  bilinear_up_and_concat( conv7, conv2, channel*2, channel*4, scope=ext+"g_up_3" )
    conv8=slim.conv2d(up8,  channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv8_1')
    conv8=slim.conv2d(conv8,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv8_2')
    up9 =  bilinear_up_and_concat( conv8, conv1, channel, channel*2, scope=ext+"g_up_4" )
    conv9=slim.conv2d(up9,  channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv9_1')
    conv9=slim.conv2d(conv9, 5,[3,3], rate=1, activation_fn=None,  scope=ext+'g_conv9_2')

    upr6 =  bilinear_up_and_concat( conv5, conv4, channel*8, channel*16, scope=ext+"g_upr_1" )
    convr6=slim.conv2d(upr6,  channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr6_1')
    convr6=slim.conv2d(convr6,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr6_2')
    upr7 =  bilinear_up_and_concat( convr6, conv3, channel*4, channel*8, scope=ext+"g_upr_2" )
    convr7=slim.conv2d(upr7,  channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr7_1')
    convr7=slim.conv2d(convr7,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr7_2')
    upr8 =  bilinear_up_and_concat( convr7, conv2, channel*2, channel*4, scope=ext+"g_upr_3" )
    convr8=slim.conv2d(upr8,  channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr8_1')
    convr8=slim.conv2d(convr8,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr8_2')
    upr9 =  bilinear_up_and_concat( convr8, conv1, channel, channel*2, scope=ext+"g_upr_4" )
    convr9=slim.conv2d(upr9,  channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr9_1')
    convr9=slim.conv2d(convr9, 5,[3,3], rate=1, activation_fn=None,  scope=ext+'g_convr9_2')

    return tf.concat([conv9,convr9],axis=3)

def net_seperate(input, channel=32, output_channel=10,reuse=False,ext=""):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    conv1=slim.conv2d(input,channel,[1,1], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_1')
    conv1=slim.conv2d(conv1,channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='same' )
    conv2=slim.conv2d(pool1,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_1')
    conv2=slim.conv2d(conv2,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='same' )
    conv3=slim.conv2d(pool2,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv3_1')
    conv3=slim.conv2d(conv3,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='same' )
    conv4=slim.conv2d(pool3,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv4_1')
    conv4=slim.conv2d(conv4,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='same' )
    conv5=slim.conv2d(pool4,channel*16,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv5_1')
    conv5=slim.conv2d(conv5,channel*16,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv5_2')


    up6 =  bilinear_up_and_concat( conv5, conv4, channel*8, channel*16, scope=ext+"g_up_1" )
    conv6=slim.conv2d(up6,  channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_1')
    conv6=slim.conv2d(conv6,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_2')
    up7 =  bilinear_up_and_concat( conv6, conv3, channel*4, channel*8, scope=ext+"g_up_2" )
    conv7=slim.conv2d(up7,  channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv7_1')
    conv7=slim.conv2d(conv7,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv7_2')
    up8 =  bilinear_up_and_concat( conv7, conv2, channel*2, channel*4, scope=ext+"g_up_3" )
    conv8=slim.conv2d(up8,  channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv8_1')
    conv8=slim.conv2d(conv8,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv8_2')
    up9 =  bilinear_up_and_concat( conv8, conv1, channel, channel*2, scope=ext+"g_up_4" )
    conv9=slim.conv2d(up9,  channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv9_1')
    conv9=slim.conv2d(conv9, 5,[3,3], rate=1, activation_fn=None,  scope=ext+'g_conv9_2')

    upr6 =  bilinear_up_and_concat( conv5, conv4, channel*8, channel*16, scope=ext+"g_upr_1" )
    convr6=slim.conv2d(upr6,  channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr6_1')
    convr6=slim.conv2d(convr6,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr6_2')
    upr7 =  bilinear_up_and_concat( convr6, conv3, channel*4, channel*8, scope=ext+"g_upr_2" )
    convr7=slim.conv2d(upr7,  channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr7_1')
    convr7=slim.conv2d(convr7,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr7_2')
    upr8 =  bilinear_up_and_concat( convr7, conv2, channel*2, channel*4, scope=ext+"g_upr_3" )
    convr8=slim.conv2d(upr8,  channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr8_1')
    convr8=slim.conv2d(convr8,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr8_2')
    upr9 =  bilinear_up_and_concat( convr8, conv1, channel, channel*2, scope=ext+"g_upr_4" )
    convr9=slim.conv2d(upr9,  channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr9_1')
    convr9=slim.conv2d(convr9, 5,[3,3], rate=1, activation_fn=None,  scope=ext+'g_convr9_2')

    return tf.concat([conv9,convr9],axis=3)

def segrrnet(input, channel=32, output_channel=10,reuse=False,ext=""):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    conv1=slim.conv2d(input,channel,[1,1], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_1')
    conv1=slim.conv2d(conv1,channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='same' )
    conv2=slim.conv2d(pool1,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_1')
    conv2=slim.conv2d(conv2,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='same' )
    conv3=slim.conv2d(pool2,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv3_1')
    conv3=slim.conv2d(conv3,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='same' )
    conv4=slim.conv2d(pool3,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv4_1')
    conv4=slim.conv2d(conv4,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='same' )
    conv5=slim.conv2d(pool4,channel*16,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv5_1')
    conv5=slim.conv2d(conv5,channel*16,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv5_2')


    up6 =  bilinear_up_and_concat( conv5, conv4, channel*8, channel*16, scope=ext+"g_up_1" )
    conv6=slim.conv2d(up6,  channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_1')
    conv6=slim.conv2d(conv6,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_2')
    up7 =  bilinear_up_and_concat( conv6, conv3, channel*4, channel*8, scope=ext+"g_up_2" )
    conv7=slim.conv2d(up7,  channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv7_1')
    conv7=slim.conv2d(conv7,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv7_2')
    up8 =  bilinear_up_and_concat( conv7, conv2, channel*2, channel*4, scope=ext+"g_up_3" )
    conv8=slim.conv2d(up8,  channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv8_1')
    conv8=slim.conv2d(conv8,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv8_2')
    up9 =  bilinear_up_and_concat( conv8, conv1, channel, channel*2, scope=ext+"g_up_4" )
    conv9=slim.conv2d(up9,  channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv9_1')
    conv9=slim.conv2d(conv9, 5,[3,3], rate=1, activation_fn=None,  scope=ext+'g_conv9_2')

    upr6 =  bilinear_up_and_concat( conv5, conv4, channel*8, channel*16, scope=ext+"g_upr_1" )
    convr6=slim.conv2d(upr6,  channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr6_1')
    convr6=slim.conv2d(convr6,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr6_2')
    upr7 =  bilinear_up_and_concat( convr6, conv3, channel*4, channel*8, scope=ext+"g_upr_2" )
    convr7=slim.conv2d(upr7,  channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr7_1')
    convr7=slim.conv2d(convr7,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr7_2')
    upr8 =  bilinear_up_and_concat( convr7, conv2, channel*2, channel*4, scope=ext+"g_upr_3" )
    convr8=slim.conv2d(upr8,  channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr8_1')
    convr8=slim.conv2d(convr8,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr8_2')
    upr9 =  bilinear_up_and_concat( convr8, conv1, channel, channel*2, scope=ext+"g_upr_4" )
    convr9=slim.conv2d(upr9,  channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convr9_1')
    convr9=slim.conv2d(convr9, 5,[3,3], rate=1, activation_fn=None,  scope=ext+'g_convr9_2')

    ups6 =  bilinear_up_and_concat( conv5, conv4, channel*8, channel*16, scope=ext+"g_ups_1" )
    convs6=slim.conv2d(ups6,  channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convs6_1')
    convs6=slim.conv2d(convs6,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convs6_2')
    ups7 =  bilinear_up_and_concat( convs6, conv3, channel*4, channel*8, scope=ext+"g_ups_2" )
    convs7=slim.conv2d(ups7,  channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convs7_1')
    convs7=slim.conv2d(convs7,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convs7_2')
    ups8 =  bilinear_up_and_concat( convs7, conv2, channel*2, channel*4, scope=ext+"g_ups_3" )
    convs8=slim.conv2d(ups8,  channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convs8_1')
    convs8=slim.conv2d(convs8,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convs8_2')
    ups9 =  bilinear_up_and_concat( convs8, conv1, channel, channel*2, scope=ext+"g_ups_4" )
    convs9=slim.conv2d(ups9,  channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_convs9_1')
    convs9=slim.conv2d(convs9, 2,[3,3], rate=1, activation_fn=None,  scope=ext+'g_convs9_2')
    return conv9, convr9, convs9

def loss(output, gt):
	return tf.reduce_mean(tf.abs(output - gt))

def l2_loss(output, gt):
    return tf.reduce_mean(tf.square(output - gt))


def tf_calculate_adolp(i1, i2, i3, i4):
    i = 0.5 * (i1 + i2 + i3 + i4)+1e-4
    q = i1 - i3 
    u = i2 - i4
    zero_mat = tf.zeros(tf.shape(i1), tf.float32)
    ones_mat = 1e-4 * tf.ones(tf.shape(i1), tf.float32)
    q = tf.where(tf.equal(q, zero_mat), ones_mat, q)
    dolp = tf.divide(tf.sqrt(tf.square(q)+tf.square(u)), i)
    aolp = 0.5*tf.atan(u/q)
    aolp = (aolp + 0.786)/(2*0.786)
    return aolp, dolp

def adolp_loss(gt, output):
	pass
	return 0

def cov_loss(p, q):
    cov_loss = tf.reduce_mean(tf.abs(tf.tanh(p-tf.reduce_mean(p))* tf.tanh(q-tf.reduce_mean(q))))
    return cov_loss

def poolcov_loss(p, q):
    pool_p = tf.nn.avg_pool(p, [1,16,16,1], [1,16,16,1], padding='valid')
    pool_q = tf.nn.avg_pool(q, [1,16,16,1], [1,16,16,1], padding='valid')
    up_p = tf.image.resize_images(p, [tf.shape(p)[1], tf.shape(p)[2]])
    up_q = tf.image.resize_images(q, [tf.shape(q)[1], tf.shape(q)[2]])
    cov_loss = tf.reduce_mean(tf.abs(tf.tanh(p-up_p)* tf.tanh(q-up_q)))
    return cov_loss



def reconstruction_loss(gt, output):
    outs = [output[:,:,:,i] for i in range(5)]
    gts  = [gt[:,:,:,i] for i in range(5)]
    i_13 = tf.abs(tf.abs(outs[0]+outs[2]) - tf.abs(gts[0] + gts[2]))
    i_24 = tf.abs(tf.abs(outs[1]+outs[3]) - tf.abs(gts[1] + gts[3]))
    loss_intra = tf.abs(outs[0]+outs[1]+outs[2]+outs[3]-2*outs[4])
    return tf.reduce_mean(i_13) + tf.reduce_mean(i_24)+0.1*tf.reduce_mean(loss_intra)
    
def compute_exclusion_loss(img1,img2,level=1):
    gradx_loss=[]
    grady_loss=[]
    
    for l in range(level):
        gradx1, grady1=compute_gradient(img1)
        gradx2, grady2=compute_gradient(img2)
        alphax=2.0*tf.reduce_mean(tf.abs(gradx1))/tf.reduce_mean(tf.abs(gradx2))#alphax=tf.reduce_mean(tf.square(gradx1))/
        alphay=2.0*tf.reduce_mean(tf.abs(grady1))/tf.reduce_mean(tf.abs(grady2))#
        
        gradx1_s=(tf.nn.sigmoid(gradx1)*2)-1
        grady1_s=(tf.nn.sigmoid(grady1)*2)-1
        gradx2_s=(tf.nn.sigmoid(gradx2*alphax)*2)-1
        grady2_s=(tf.nn.sigmoid(grady2*alphay)*2)-1

        gradx_loss.append(tf.reduce_mean(tf.multiply(tf.square(gradx1_s),tf.square(gradx2_s)),reduction_indices=[1,2,3])**0.25)
        grady_loss.append(tf.reduce_mean(tf.multiply(tf.square(grady1_s),tf.square(grady2_s)),reduction_indices=[1,2,3])**0.25)

        img1=tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='same')
        img2=tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='same')
    return gradx_loss,grady_loss


###################
##
##################

# def compute_exclusion_loss(img1,img2,level=1):
#     gradx_loss=[]
#     grady_loss=[]
    
#     for l in range(level):
#         gradx1, grady1=compute_gradient(img1)
#         gradx2, grady2=compute_gradient(img2)
#         alphax=2.0*tf.reduce_mean(tf.abs(gradx1))/tf.reduce_mean(tf.abs(gradx2))#alphax=tf.reduce_mean(tf.square(gradx1))/
#         alphay=2.0*tf.reduce_mean(tf.abs(grady1))/tf.reduce_mean(tf.abs(grady2))#
        
#         gradx1_s=(tf.nn.sigmoid(gradx1)*2)-1
#         grady1_s=(tf.nn.sigmoid(grady1)*2)-1
#         gradx2_s=(tf.nn.sigmoid(gradx2*alphax)*2)-1
#         grady2_s=(tf.nn.sigmoid(grady2*alphay)*2)-1

#         gradx_loss.append(tf.reduce_mean(tf.multiply(tf.square(gradx1_s),tf.square(gradx2_s)),reduction_indices=[1,2,3])**0.25)
#         grady_loss.append(tf.reduce_mean(tf.multiply(tf.square(grady1_s),tf.square(grady2_s)),reduction_indices=[1,2,3])**0.25)

#         img1=tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='same')
#         img2=tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='same')
#     return gradx_loss,grady_loss

def l1_gradient_reg(img1, img2):
	img1, img2=0.5*img1, 0.5*img2
	img1_gradx, img1_grady= compute_gradient(img1)
	img2_gradx, img2_grady= compute_gradient(img2)
	img1_reg = tf.reduce_mean(tf.sqrt(tf.square(img1_gradx[:,:,:-1,:]) +  tf.square(img1_grady[:,:-1,:,:])))
	img2_reg = tf.reduce_mean(tf.sqrt(tf.square(img2_gradx[:,:,:-1,:]) +  tf.square(img2_grady[:,:-1,:,:])))
	return img1_reg + img2_reg

def l2_gradient_exclu(img1, img2):
	img1, img2=0.5*img1, 0.5*img2
	img1_gradx, img1_grady= compute_gradient(img1)
	img2_gradx, img2_grady= compute_gradient(img2)
	img1_grad = tf.square(img1_gradx[:,:,:-1,:])+tf.square(img1_grady[:,:-1,:,:])
	img2_grad = tf.square(img2_gradx[:,:,:-1,:])+tf.square(img2_grady[:,:-1,:,:])
	loss = tf.reduce_mean(img1_grad * img2_grad)
	return loss


def compute_gradient(img):
    gradx=img[:,1:,:,:]-img[:,:-1,:,:]
    grady=img[:,:,1:,:]-img[:,:,:-1,:]
    return gradx,grady

def paper_exclusion_loss(img1,img2,level=3):
    gradx_loss=[]
    grady_loss=[]
    
    for l in range(level):
        gradx1, grady1=compute_gradient(img1)
        gradx2, grady2=compute_gradient(img2)
        alphax=tf.reduce_sqrt(tf.reduce_mean(tf.square(gradx1)))/tf.reduce_sqrt(tf.reduce_mean(tf.square(gradx2)))
        alphay=tf.reduce_sqrt(tf.reduce_mean(tf.square(grady1)))/tf.reduce_sqrt(tf.reduce_mean(tf.square(grady2)))
        
        gradx1_s=(tf.nn.sigmoid(gradx1)*2)-1
        grady1_s=(tf.nn.sigmoid(grady1)*2)-1
        gradx2_s=(tf.nn.sigmoid(gradx2*alphax)*2)-1
        grady2_s=(tf.nn.sigmoid(grady2*alphay)*2)-1

        gradx_loss.append(tf.reduce_mean(tf.multiply(tf.square(gradx1_s),tf.square(gradx2_s)),reduction_indices=[1,2,3])**0.25)
        grady_loss.append(tf.reduce_mean(tf.multiply(tf.square(grady1_s),tf.square(grady2_s)),reduction_indices=[1,2,3])**0.25)

        img1=tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='same')
        img2=tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='same')
    return tf.reduce_mean(gradx_loss) + tf.reduce_mean(grady_loss)
def gcn(input, channel=32, output_channel=2,reuse=False,ext=""):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    conv1=slim.conv2d(input,channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_1')
    conv1=slim.conv2d(conv1,channel,[3,3], rate=1, activation_fn=lrelu,scope=ext+'g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2])
    conv2=slim.conv2d(pool1,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_1')
    conv2=slim.conv2d(conv2,channel*2,[3,3], rate=1, activation_fn=lrelu,scope=ext+'g_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2])
    conv3=slim.conv2d(pool2,channel*4,[3,3], rate=1, activation_fn=lrelu,scope=ext+'g_conv3_1')
    conv3=slim.conv2d(conv3,channel*4,[3,3], rate=1, activation_fn=lrelu,scope=ext+'g_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2])
    conv4=slim.conv2d(pool3,channel*8,[3,3], rate=1, activation_fn=lrelu,scope=ext+'g_conv4_1')
    conv4=slim.conv2d(conv4,channel*8,[3,3], rate=1, activation_fn=lrelu,scope=ext+'g_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2])
    conv5=slim.conv2d(pool4,channel*16,[3,3], rate=1, activation_fn=lrelu,scope=ext+'g_conv5_1')
    conv5=slim.conv2d(conv5,channel*16,[3,3], rate=1, activation_fn=lrelu,scope=ext+'g_conv5_2')
    pool5=slim.max_pool2d(conv5, [2, 2])
    pool5=slim.flatten(pool5)
    fc6 = slim.fully_connected(pool5, 4096, activation_fn=lrelu, scope= "fc6")
    fc7 = slim.fully_connected(fc6, 1024, activation_fn=lrelu, scope="fc7")
    fc8 = slim.fully_connected(fc7, 2, activation_fn=None, scope="fc8")
    #prob = tf.nn.softmax(fc8, name="prob")
    return fc8 

def vgg16(inputs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],activation_fn=tf.nn.relu, weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv6')
        net = slim.flatten(net) 
        net = slim.fully_connected(net, 4096, scope='fc6')
        net = slim.dropout(net, 0.5, scope='dropout6')
        net = slim.fully_connected(net, 4096, scope='fc7')
        net = slim.dropout(net, 0.5, scope='dropout7')
        net = slim.fully_connected(net, 2, activation_fn=None, scope='fc8')
        return net

