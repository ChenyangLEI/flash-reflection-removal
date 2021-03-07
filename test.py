from __future__ import division
import os
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import argparse
import subprocess
from scipy.misc import imread, imsave
from model.network import UNet as UNet
from model.network import UNet_SE as UNet_SE
from glob import glob
import random
from tqdm import tqdm

seed = 2019
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="full_global_lp",help="path to folder containing the model")
parser.add_argument("--testset", default="real",help="path to folder containing the model")
ARGS = parser.parse_args()
model=ARGS.model

continue_training=True
os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax([int(x.split()[2]) for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
os.environ["OMP_NUM_THREADS"] = '4'
print(ARGS)

def detect_shadow(ambient, flashonly):
    intensity_ambient = tf.norm(ambient, axis=3, keepdims=True)
    intensity_flashonly = tf.norm(flashonly, axis=3, keepdims=True)
    ambient_ratio = intensity_ambient / tf.reduce_mean(intensity_ambient)
    flashonly_ratio = intensity_flashonly / tf.reduce_mean(intensity_flashonly)
    
    # Dark in PF but not dark in F
    pf_div_by_ambient = flashonly_ratio / (ambient_ratio+1e-5)
    shadow_mask = tf.cast(tf.less(pf_div_by_ambient, 0.8), tf.float32)
    
    # Cannot be too bright in flashonly
    dark_mask = tf.cast(tf.less(intensity_flashonly, 0.3), tf.float32)
    mask = dark_mask * shadow_mask
    return mask


# exit()

# set up the model and define the graph
lossDict= {}
with tf.variable_scope(tf.get_variable_scope()):
    input_ambient=tf.placeholder(tf.float32,shape=[None,None,None,3])
    input_pureflash=tf.placeholder(tf.float32,shape=[None,None,None,3])
    input_flash=tf.placeholder(tf.float32,shape=[None,None,None,3])
    mask_shadow = tf.cast(tf.greater(input_pureflash, 0.02), tf.float32)
    mask_highlight = tf.cast(tf.less(input_flash, 0.96), tf.float32)
    mask_shadow_highlight = mask_shadow * mask_highlight
    gray_pureflash = 0.33 * (input_pureflash[...,0:1] + input_pureflash[...,1:2] + input_pureflash[...,2:3])
    bad_mask = detect_shadow(input_ambient, input_pureflash)
    reflection_layer = UNet_SE(tf.concat([input_ambient, gray_pureflash, (-bad_mask + 1)], axis=3), output_channel = 3, ext='Ref_')
    transmission_layer = UNet_SE(tf.concat([input_ambient, reflection_layer, (-bad_mask + 1)], axis=3), ext='Trans_')



######### Session #########
saver = tf.train.Saver(max_to_keep=20)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
var_restore = [v for v in tf.trainable_variables()]
saver_restore = tf.train.Saver(var_restore)
ckpt = tf.train.get_checkpoint_state('./ckpt/'+model)
######### Session #########


print("[i] contain checkpoint: ", ckpt)
if ckpt and continue_training:
    saver_restore=tf.train.Saver([var for var in tf.trainable_variables()])
    print('loaded '+ckpt.model_checkpoint_path)
    saver_restore.restore(sess,ckpt.model_checkpoint_path)


data_dir = "./data/{}".format(ARGS.testset)
data_names = sorted(glob(data_dir+"/*ambient.jpg"))

def crop_shape(tmp_all, size=32):
    h,w = tmp_all.shape[1:3]
    h = h // size * size
    w = w // size * size
    return h, w

num_test = len(data_names)
print(num_test)
for epoch in range(9999, 10000):
    print("Processing epoch %d"%epoch, "./ckpt/%s/%s"%(model,data_dir.split("/")[-2]))
    # save model and images every epoch
    save_dir = "./ckpt/%s/%s"%(model,data_dir.split("/")[-2])
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    print("output path: {}".format(save_dir))
    all_loss_test=np.zeros(num_test, dtype=float)
    metrics = {"T_ssim":0,"T_psnr":0,"R_ssim":0, "R_psnr":0}
    fetch_list=[transmission_layer, reflection_layer, input_ambient, input_flash, input_pureflash, bad_mask]
    for id in tqdm(range(num_test)):
        st=time.time()
        tmp_pureflash = imread(data_names[id].replace("ambient.jpg", "pureflash.jpg"))[None,...] / 255.
        tmp_ambient = imread(data_names[id])[None,...] / 255.
        tmp_flash = imread(data_names[id].replace("ambient.jpg", "flash.jpg"))[None,...] / 255.
        h,w = crop_shape(tmp_ambient, size=32)
        tmp_ambient, tmp_pureflash, tmp_flash = tmp_ambient[:,:h,:w,:], tmp_pureflash[:,:h,:w,:], tmp_flash[:,:h,:w,:]
        pred_image_t, pred_image_r, in_ambient, in_flash, in_pureflash, pred_mask = sess.run(fetch_list,
            feed_dict={input_ambient:tmp_ambient, input_pureflash:tmp_pureflash, input_flash: tmp_flash})
        # print("Epc: %3d, shape of outputs: "%epoch, pred_image_t.shape, pred_image_r.shape)
        save_path = "{}/{}".format(save_dir, data_names[id].split("/")[-1])
        imsave(save_path.replace("ambient.jpg", "_0_input_ambient.png"), np.uint8(tmp_ambient[0].clip(0,1) * 255.))
        imsave(save_path.replace("ambient.jpg", "_1_pred_transmission.png"), np.uint8(pred_image_t[0].clip(0,1) * 255.))
        # imsave(save_path.replace("ambient.jpg", "_2_pred_refletion.png"), np.uint8(pred_image_r[0].clip(0,1) * 255.))
        imsave(save_path.replace("ambient.jpg", "_3_input_flash.png"), np.uint8(tmp_flash[0].clip(0,1) * 255.))
        imsave(save_path.replace("ambient.jpg", "_4_input_pureflash.png"), np.uint8(tmp_pureflash[0].clip(0,1) * 255.))
        # imsave(save_path.replace("ambient.jpg", "_5_mask.png"), np.uint8(pred_mask[0,...,0].clip(0,1) * 255.))

