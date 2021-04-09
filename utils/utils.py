from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,time,cv2,scipy.io
import tensorflow as tf
# import tensorflow.contrib.slim as slim
import scipy.misc as sic
# import network as network
import subprocess
import numpy as np
from matplotlib.colors import hsv_to_rgb
from skimage.measure import compare_ssim, compare_psnr
from glob import glob



def prepare_data(data_path='../data_new/Data_Polar_Clean/crop_npy/'):
    train_items, val_items = [], []
    folders1 = glob(data_path+'/*')
#    print(folders1)
    folders2 = []
    for folder1 in folders1:
        folders2 = folders2 + glob(folder1+'/Indoor/*') + glob(folder1+'/Outdoor/*')
#    print(folders2) 
    folders2.sort()   
    for folder2 in folders2[1::5] + folders2[2::5]+folders2[3::5]+folders2[4::5]:
        folder = folder2
        imgs = glob(folder + '/*.npy')
        imgs.sort()
     #   print(folder, len(imgs))
        for idx in range(len(imgs)//2):
            tmp_M = imgs[2*idx+1] 
            tmp_R = imgs[2*idx]
            train_items.append([tmp_M,tmp_R])
      #      print(tmp_R, tmp_M)

    for folder2 in folders2[::5]: 
        folder = folder2
        imgs = glob(folder + '/*.npy')
        imgs.sort()
        print(folder, len(imgs))
        for idx in range(len(imgs)//2):
            tmp_M = imgs[2*idx+1] 
            tmp_R = imgs[2*idx]
            val_items.append([tmp_M,tmp_R])
       #     print(tmp_R, tmp_M)
    return train_items, val_items[::3]

def prepare_final_data(data_path='../data_new/Data_Polar_Clean/crop_npy/'):
    train_items,val_items,test_items=[],[],[]
    imgs = glob("../data_new/Data_Polar_Clean/MMR_1/train/*npy")
    imgs.sort()
    for idx in range(len(imgs)//2):
        tmp_M = imgs[2*idx+1] 
        tmp_R = imgs[2*idx]
        train_items.append([tmp_M,tmp_R])
      #      print(tmp_R, tmp_M)

    imgs = glob("../data_new/Data_Polar_Clean/MMR_1/test/*npy")
    imgs.sort()
    for idx in range(len(imgs)//2):
        tmp_M = imgs[2*idx+1] 
        tmp_R = imgs[2*idx]
        test_items.append([tmp_M,tmp_R])
      #      print(tmp_R, tmp_M)

    imgs = glob("../data_new/Data_Polar_Clean/MMR_1/val/*npy")
    imgs.sort()
    for idx in range(len(imgs)//2):
        tmp_M = imgs[2*idx+1] 
        tmp_R = imgs[2*idx]
        val_items.append([tmp_M,tmp_R])
      #      print(tmp_R, tmp_M)

    return train_items, val_items, test_items

def prepare_item(item):
    M_name, R_name = item
    tmp_M = np.load(M_name)
    tmp_R = np.load(R_name)
    return tmp_M,tmp_R 

def light_mask(h, w):
    mid_h = h//5 + np.random.randint(h//5*3)
    mid_w = w//5 + np.random.randint(w//5*3)
    Light_low = 0.1+0.3*np.random.random()
    Light_high= Light_low + 1*np.random.random()
    row2 = np.concatenate([np.linspace(Light_low,0.8,mid_w),np.linspace(0.8,Light_low, w-mid_w)],axis=0)
    mat2 = np.tile(row2[np.newaxis,:],[h,1])
    row1 = np.concatenate([np.linspace(Light_low,0.8,mid_h),np.linspace(0.8,Light_low, h-mid_h)],axis=0)
    mat1 = np.tile(row1[:,np.newaxis],[1,w])
    mat = np.power(mat1*mat2, 2)
    # mat = np.power(mat, 1/2.2)
    sz = (20 + np.random.randint(20))*2 + 1
    mask1=cv2.GaussianBlur(mat,(sz,sz),cv2.BORDER_DEFAULT)
    return mask1

def shadow_mask(img):                                                        
    h_orig,w_orig = img.shape[:2]
    mask = np.ones((h_orig, w_orig))
    w_crop = np.random.randint(10, w_orig//3)
    h_crop = np.random.randint(10, h_orig//3)
    try:                                                                       
        w_offset = np.random.randint(0, w_orig-w_crop-1) 
        h_offset = np.random.randint(0, h_orig-h_crop-1) 
    except:  
        print("Original W %d, desired W %d"%(w_orig,w_crop))                  
        print("Original H %d, desired H %d"%(h_orig,h_crop))              
    print(mask.shape)
    mask[h_offset:h_offset+h_crop-1,w_offset:w_offset+w_crop-1] = 0.2 + 0.4*np.random.rand()
    
    w_crop = np.random.randint(10, w_orig//3)
    h_crop = np.random.randint(10, h_orig//3)
    try:                                                                       
        w_offset = np.random.randint(0, w_orig-w_crop-1) 
        h_offset = np.random.randint(0, h_orig-h_crop-1) 
    except:  
        print("Original W %d, desired W %d"%(w_orig,w_crop))                  
        print("Original H %d, desired H %d"%(h_orig,h_crop))              
    print(mask.shape)
    mask[h_offset:h_offset+h_crop-1,w_offset:w_offset+w_crop-1] = 0.3 + 0.4*np.random.rand()

    
    w_crop = np.random.randint(10, w_orig//3)
    h_crop = np.random.randint(10, h_orig//3)
    try:                                                                       
        w_offset = np.random.randint(0, w_orig-w_crop-1) 
        h_offset = np.random.randint(0, h_orig-h_crop-1) 
    except:  
        print("Original W %d, desired W %d"%(w_orig,w_crop))                  
        print("Original H %d, desired H %d"%(h_orig,h_crop))              
    print(mask.shape)
    mask[h_offset:h_offset+h_crop-1,w_offset:w_offset+w_crop-1] = 0.4 + 0.4*np.random.rand()

    return mask

def prepare_FNF(item):
    #---------------Get R&T----------------$
    T_name, R_name = item
    syn_image1=cv2.imread(T_name,-1)
    w=np.random.randint(256, 480)
    h=round((w/syn_image1.shape[1])*syn_image1.shape[0])
    t=cv2.resize(np.float32(syn_image1),(w,h),cv2.INTER_CUBIC)/255.0
    r=cv2.resize(np.float32(cv2.imread(R_name,-1)),(w,h),cv2.INTER_CUBIC)/255.0
    # h, w = nf.shape[:2]


    alpha = 0.25 + 0.5*np.random.random()
    gt_r = r
    gt_r = (1-alpha)*gt_r
    gt_t = alpha * t

    nf = np.power(np.power(gt_t,2.2) + np.power(gt_r,2.2), 1/2.2)

    pf = (0.5+ 0.7*np.random.random()) * t #

    mask1= light_mask(h,w)
    mask2= light_mask(h,w)
    mask = np.sqrt(mask1*mask2)

    shadow = shadow_mask(pf)
    if np.random.random() < 0.5:
        pf = np.power(np.power(pf,2.2) + 0.5* mask[:,:,np.newaxis],1/2.2)
    else:
        pf = np.power(np.power(pf,2.2) * mask[:,:,np.newaxis],1/2.2)
    pf = pf*shadow[:,:,np.newaxis]
    h = h//32 * 32
    w = w//32 * 32

    return pf[np.newaxis, :h, :w, :], gt_t[np.newaxis, :h, :w, :],gt_r[np.newaxis, :h, :w, :],nf[np.newaxis, :h, :w, :]

def get_metrics(metrics,out_mask, gt_target,gt_reflection,pred_image_t,pred_image_r):
    metrics["T_ssim"] += compare_ssim(0.5*gt_target[0,:,:,4]*out_mask[0,:,:,0], 0.5*pred_image_t[0,:,:,4]*out_mask[0,:,:,0])
    metrics["T_psnr"] += compare_psnr(0.5*gt_target[0,:,:,4]*out_mask[0,:,:,0], 0.5*pred_image_t[0,:,:,4]*out_mask[0,:,:,0], 1)
    metrics["R_ssim"] += compare_ssim(0.5*gt_reflection[0,:,:,4]*out_mask[0,:,:,0], 0.5*pred_image_r[0,:,:,4]*out_mask[0,:,:,0])
    metrics["R_psnr"] += compare_psnr(0.5*gt_reflection[0,:,:,4]*out_mask[0,:,:,0], 0.5*pred_image_r[0,:,:,4]*out_mask[0,:,:,0], 1)
    return metrics

def save_concat_img(gt_input, gt_target, gt_reflection, pureflash, pred_image_t, pred_image_r, save_path, in_flash=None, is_test=False):
    if is_test == True:
        sic.imsave(save_path.replace(".jpg", "_0_input_ambient.jpg"), np.uint8(gt_input[0].clip(0,1) * 255.))
        sic.imsave(save_path.replace(".jpg", "_5_input_flash.jpg"), np.uint8(in_flash[0].clip(0,1) * 255.))
        sic.imsave(save_path.replace(".jpg", "_6_input_pureflash.jpg"), np.uint8(pureflash[0].clip(0,1) * 255.))
        sic.imsave(save_path.replace(".jpg", "_1_pred_transmission.jpg"), np.uint8(pred_image_t[0].clip(0,1) * 255.))
        sic.imsave(save_path.replace(".jpg", "_2_pred_refletion.jpg"), np.uint8(pred_image_r[0].clip(0,1) * 255.))
        sic.imsave(save_path.replace(".jpg", "_3_gt_transmission.jpg"), np.uint8(gt_target[0].clip(0,1) * 255.))
        sic.imsave(save_path.replace(".jpg", "_4_gt_reflection.jpg"), np.uint8(gt_reflection[0].clip(0,1) * 255.))
        return 0


    # out_img1= np.concatenate([gt_input[0], gt_target[0], gt_reflection[0]], axis=1)
    h, w = gt_input.shape[1:3]
    out_img1 = [gt_input[0], pred_image_t[0],  gt_target[0]]
    names = ["Input", "Pred", "GT"]
    for idx, img in enumerate(out_img1):
        cv2.putText(img, names[idx], (w//2-len(names[idx])*10, h-20), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
    out_img1 = np.hstack(out_img1) 


    # out_img2= np.concatenate([pureflash[0], pred_image_t[0],pred_image_r[0]], axis=1)
    out_img2= [pureflash[0], gt_reflection[0], pred_image_r[0]]

    # names = ["I_fo", "pred_R_a", "R_a"]
    # for idx, img in enumerate(out_img2):
    #     print(img.shape)
    #     cv2.putText(img, names[idx], (w//2-len(names[idx])*10, h-20), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
    out_img2 = np.hstack(out_img2)     
    out_img = np.vstack([out_img1, out_img2])

    out_img = np.minimum(np.maximum(out_img,0.0),1.0)*255.0
    # cv2.imwrite("result/%s/%04d/val_%06d.jpg"%(task, epoch, id), np.uint8(out_img[::2,::2]))
    # cv2.imwrite(save_path, np.uint8(out_img[::2,::2]))
    sic.imsave(save_path, np.uint8(out_img[::2,::2]))
    return out_img

def save_results(all_loss_test, metrics, id, task,epoch):
    result=open("result/%s/score.txt"%task,'a')
    result.write("Epc: %03d Loss: %.5f | SSIM: %.3f PSNR: %.2f | SSIM: %.3f PSNR: %.2f \n"%\
        (epoch, np.mean(all_loss_test[np.where(all_loss_test)]), metrics["T_ssim"]/(id+1), metrics["T_psnr"]/(id+1), metrics["R_ssim"]/(id+1), metrics["R_psnr"]/(id+1)))
    result.close()

def crop_shape(tmp_all, size=32):
    h,w = tmp_all.shape[1:3]
    h = h // size * size
    w = w // size * size
    return h, w

def cnts_add_display(epoch, cnts, step,crt, crt_t, st):
    cnts["cnt"]+=1
    step+=1
    cnts["all_r"] += crt
    cnts["all_t"] += crt_t
    
    cnt, all_r, all_t = cnts["cnt"],cnts["all_r"],cnts["all_t"]
    print("iter: %03d %03d %d || r:%.3f  %.3f | t:%.3f  %.3f |time:%.2f"%\
        (epoch,cnt,step,crt,all_r/cnt,crt_t,all_t/cnt,time.time()-st))    
    return cnts, step


def save_all_out(output, path_prefix, HSV=0, I14=0,AoLP=0,DoLP=0):
    sic.imsave("%s_I.jpg"%path_prefix,np.uint8(np.maximum(np.minimum(output[0,:,:,4]*255.0,255.0),0.0)))
    if I14:
        sic.imsave("%s_I14.jpg"%path_prefix,np.uint8(np.maximum(np.minimum(np.concatenate([output[0,:,:,i] for i in range(4)],axis=0)*255.0,255.0),0.0)))
    if HSV:
        sic.imsave("%s_HSV.jpg"%path_prefix,np.uint8(np.maximum(np.minimum(output[0,:,:,-3:]*255.0,255.0),0.0)))
    if AoLP:
        sic.imsave("%s_AoLP.jpg"%path_prefix,np.uint8(np.maximum(np.minimum(output[0,:,:,6]*255.0,255.0),0.0)))
    if DoLP:
        sic.imsave("%s_DoLP.jpg"%path_prefix,np.uint8(np.maximum(np.minimum(output[0,:,:,5]*255.0,255.0),0.0)))

def get_input(path, id):
    raw_in_name = path + '/in/%04d.png'%id
    raw_outR_name=path + '/out/%04d.png'%id
    raw_outT_name=path + '/out/%04d.png'%id
    temp_input = get_from_raw(raw_in_name, raw=None)
    temp_output= np.concatenate([temp_input, temp_input],axis=3)
    # temp_output= np.concatenate([get_from_raw(raw_outR_name,raw=None), get_from_raw(raw_outT_name,raw=None)],axis=3)
    # temp_output=None

    return temp_input, temp_output

def load_data(train_path, test_path, train_num, test_num):
    train_in = []
    test_in  = []
    train_out= []
    test_out = []
    for i in range(train_num):
        temp_input, temp_output = get_input(train_path, i+1)
        print('Train: ', i, temp_input.shape, temp_output.shape)
        
        train_in.append(temp_input)
        train_out.append(temp_output)

    for i in range(test_num):   
        temp_input, temp_output = get_input(test_path, i+1)
        print('Test: ', i, temp_input.shape, temp_output.shape)
        test_in.append(temp_input)
        test_out.append(temp_output)

    return train_in, train_out, test_in, test_out

def get_from_raw(raw_name, raw=True):
    if raw:
        raw_img = read_raw(raw_name)
    else:
        raw_img = sic.imread(raw_name, mode='L')/255.
#    print(np.mean(raw_img))
#    print(raw_name, raw_img.shape)

    h=raw_img.shape[0]//32*32
    w=raw_img.shape[1]//32*32
    return raw_split(raw_img[:h,:w])
    # return raw2imgs(raw_img[:h,:w])

def raw2imgs(raw_img):
    I3=raw_img[::2,::2] 
    I2=raw_img[::2,1::2] 
    I4=raw_img[1::2,::2] 
    I1=raw_img[1::2,1::2]
#    I3=raw_img[::2,::2][::2,::2]
#    I2=raw_img[::2,1::2][::2,::2]
#    I4=raw_img[1::2,::2][::2,::2] 
#    I1=raw_img[1::2,1::2][::2,::2]
    I = 0.5*(I1 + I3 + I2 + I4)
    #print('I1: ', I1[np.isnan(I1)].shape)
    ##print('I2: ', I1[np.isnan(I2)].shape)
    #print('I3: ', I1[np.isnan(I3)].shape)
    #print('I4: ', I1[np.isnan(I4)].shape)
    #print('I: ', I1[np.isnan(I)].shape, np.max(I), np.mean(I))
    return I1, I2, I3, I4, I

def raw_split(raw_img):
    I1, I2, I3, I4, I = raw2imgs(raw_img)
    AoLP, DoLP=calculate_ADoLP(I1, I2, I3, I4, I)
    I_p,  I_np=I * DoLP, I*(1-DoLP)
#    print('AoLP NaN: ', AoLP[np.isnan(AoLP)].shape)
#    print('DoLP NaN: ', DoLP[np.isnan(DoLP)].shape)
#    print('I_p NaN: ', I_p[np.isnan(I_p)].shape)
#    print('I_np NaN: ', I_np[np.isnan(I_np)].shape)
    data = [I1, I2, I3, I4, I, DoLP, AoLP, I_p, I_np]
    data_expand = [I[np.newaxis, :,:,np.newaxis] for I in data]
    return np.concatenate(data_expand,axis=3)

def pols2infor(raw_img):
    I1, I2, I3, I4 = [raw_img[:,:,i] for i in range(4)]
    I = (I1+I2+I3+I4)*0.5
    AoLP, DoLP=calculate_ADoLP(I1, I2, I3, I4, I)
    I_p,  I_np=I * DoLP, I*(1-DoLP)
    # print('AoLP NaN: ', AoLP[np.isnan(AoLP)].shape)
    # print('DoLP NaN: ', DoLP[np.isnan(DoLP)].shape)
    # print('I_p NaN: ', I_p[np.isnan(I_p)].shape)
    # print('I_np NaN: ', I_np[np.isnan(I_np)].shape)
    data = [I1, I2, I3, I4, I, DoLP, AoLP, I_p, I_np]
    data_expand = [I[np.newaxis, :,:,np.newaxis] for I in data]
    return np.concatenate(data_expand,axis=3)

def calculate_ADoLP(I1, I2, I3, I4, I):
    Q = I1 - I3 
    U = I2 - I4
    Q[Q == 0] = 0.0001
    I[I == 0] = 0.0001
    DoLP = np.sqrt(np.square(Q)+np.square(U))/I
    AoLP = 0.5*np.arctan(U/Q)
    # print(np.min(DoLP), np.max(DoLP))
#    AoLP = (AoLP + 0.786)/(2*0.786)
    DoLP[DoLP>1] = 1 
    return AoLP, DoLP
'''
def ad_new(raw):
    Q = raw[:,:,:,0:1] - raw[:,:,:,2:3] 
    U = raw[:,:,:,1:2] - raw[:,:,:,3:4]
    Q[Q == 0] = 1e-7 
    DoLP = np.sqrt(np.square(Q)+np.square(U))/raw[:,:,;,4:5]
    AoLP = 0.5*np.arctan(U/Q)
#    AoLP = (AoLP + 0.786)/(2*0.786)
    return np.concatenate([raw, AoLP, DoLP],axis=3)
'''

def vis_ADoLP(AoLP, DoLP):
    hsv = np.concatenate([AoLP[:,:,np.newaxis], DoLP[:,:,np.newaxis], np.ones([AoLP.shape[0], AoLP.shape[1], 1])],axis=2)
    rgb = hsv_to_rgb(hsv)
    return   rgb

def vis_ADI(raw):
    AoLP, DoLP, I=raw[:,:,2],raw[:,:,1],raw[:,:,0]
    hsv = np.concatenate([AoLP[:,:,np.newaxis], DoLP[:,:,np.newaxis], I[:,:,np.newaxis]],axis=2)
    rgb = hsv_to_rgb(hsv)
    return  rgb


def read_uint12_12p(path):
    data = np.fromfile(path, dtype=np.uint8).astype("float32")
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = fst_uint8 + (np.bitwise_and((mid_uint8 << 8), 3840))
    snd_uint12 = (lst_uint8 << 4) + (mid_uint8 >> 4)
    return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])
    
def read_raw(path, imageSize = (2048, 2448)):
    npimg = np.fromfile(path, dtype=np.uint16).astype("float32")
    unit = float(npimg.shape[0])/(2048*2448)
    if unit == 1:
        if np.max(npimg)>4096:
            npimg /= 65535.
        else:
            npimg /= 4095.
    elif unit== 0.5 :
        npimg = np.fromfile(path, dtype=np.uint8).astype("float32")
        npimg /= 255.
    else:
        npimg = np.float32(read_uint12_12p(path))/4095
    npimg = npimg.reshape(imageSize)
#     print("Normalize- Max: %.4lf Min: %.4lf"%(np.max(npimg), np.min(npimg)))
    return npimg

def whole_split(net_out):
    key = 'I1, I2, I3, I4, I, DoLP, AoLP, I_p, I_np'
    key = key.split(', ')
    data_dict = {}
    for i in range(9):
        data_dict[key[i]] = net_out[0,:,:,i]
    return data_dict

def pols2difs(raw_img):
    I1, I2, I3, I4 = [raw_img[:,:,i] for i in range(4)]
    I = (I1+I2+I3+I4)*0.5
    AoLP, DoLP=calculate_ADoLP(I1, I2, I3, I4, I)
    I_p,  I_np=I * DoLP, I*(1-DoLP)
    # print('AoLP NaN: ', AoLP[np.isnan(AoLP)].shape)
    # print('DoLP NaN: ', DoLP[np.isnan(DoLP)].shape)
    # print('I_p NaN: ', I_p[np.isnan(I_p)].shape)
    # print('I_np NaN: ', I_np[np.isnan(I_np)].shape)
    
    data = [I1, I2, I3, I4, I, DoLP, AoLP, I_p, I_np, I1-I2,  I1-I3,  I1-I4, I2-I3, I2-I4, I3-I4]
    data_expand = [I[np.newaxis, :,:,np.newaxis] for I in data]
    return np.concatenate(data_expand,axis=3)
def mask(img):
    h, w = img.shape[0], img.shape[1]
    mask = np.zeros([h, w, 1])
    x1 = np.random.randint(int(0.75*w))
    x2 = x1 + int(0.25*w)+np.random.randint(int(0.75*w - x1))
    y1 = np.random.randint(int(0.75*h))
    y2 = y1 + int(0.25*h)+np.random.randint(int(0.75*h - y1))
    mask[x1:x2, y1:y2, :] = 1
#    print("x1, x2, y1, y2: ", x1, x2, y1, y2)
    return mask

def crop_images(X,a,b,is_sq=False):
    h_orig,w_orig = X.shape[1:3]
    w_crop = np.random.randint(a, b)
    r = w_crop/w_orig
    h_crop = np.int(h_orig*r)
    try:
        w_offset = np.random.randint(0, w_orig-w_crop-1)
        h_offset = np.random.randint(0, h_orig-h_crop-1)
    except:
        print("Original W %d, desired W %d"%(w_orig,w_crop))
        print("Original H %d, desired H %d"%(h_orig,h_crop))
    out = X[:,h_offset:h_offset+h_crop-1,w_offset:w_offset+w_crop-1,:]
    h,w=out.shape[1:3]
    h = h//32*32
    w = w//32*32
    return out[:,:h,:w,:]
def aug_ad(im_in, im_R, im_T):
    #Crop
    h_orig,w_orig = im_in.shape[1:3]
    w_crop = np.random.randint(512, 641)
    r = w_crop/w_orig
    h_crop = np.int(h_orig*r)
    h_crop = h_crop//32*32
    w_crop = w_crop//32*32
    
    try:
        w_offset = np.random.randint(0, w_orig-w_crop-1)
        h_offset = np.random.randint(0, h_orig-h_crop-1)
    except:
        print("Original W %d, desired W %d"%(w_orig,w_crop))
        print("Original H %d, desired H %d"%(h_orig,h_crop))
    im_in=im_in[:,h_offset:h_offset+h_crop,w_offset:w_offset+w_crop,:]
    im_R = im_R[:,h_offset:h_offset+h_crop,w_offset:w_offset+w_crop,:]
    im_T = im_T[:,h_offset:h_offset+h_crop,w_offset:w_offset+w_crop,:]
    
    return ad_new(im_in), ad_new(im_R), ad_new(im_T)
def augmentation(im_in, im_R, im_T):
    #Crop
    h_orig,w_orig = im_in.shape[1:3]
#    w_crop = 641#np.random.randint(640, 801)
#    r = w_crop/w_orig
#    h_crop = np.int(h_orig*r)
    w_crop, h_crop = 512, 512
#    h_crop = h_crop//32*32
#    w_crop = w_crop//32*32
    
    try:
        w_offset = np.random.randint(0, w_orig-w_crop-1)
        h_offset = np.random.randint(0, h_orig-h_crop-1)
    except:
        print("Original W %d, desired W %d"%(w_orig,w_crop))
        print("Original H %d, desired H %d"%(h_orig,h_crop))
    im_in=im_in[:,h_offset:h_offset+h_crop,w_offset:w_offset+w_crop,:]
    im_R = im_R[:,h_offset:h_offset+h_crop,w_offset:w_offset+w_crop,:]
    im_T = im_T[:,h_offset:h_offset+h_crop,w_offset:w_offset+w_crop,:]
    return im_in, im_R, im_T
def crop_augmentation(im_R, im_T):
    #Crop
    h_orig,w_orig = im_R.shape[1:3]

    h_crop = h_orig//224*224
    w_crop = w_orig//224*224
    size = min(h_crop,w_crop)
#    w_crop = 641#np.random.randint(640, 801)
#    r = w_crop/w_orig
#    h_crop = np.int(h_orig*r)
    if size > 640:
        size = 640
    w_crop = size
    h_crop = size
    
    try:
        w_offset = np.random.randint(0, w_orig-w_crop)
        h_offset = np.random.randint(0, h_orig-h_crop)
    except:
        print("Original W %d, desired W %d"%(w_orig,w_crop))
        print("Original H %d, desired H %d"%(h_orig,h_crop))
    im_R = im_R[:,h_offset:h_offset+h_crop,w_offset:w_offset+w_crop,:]
    im_T = im_T[:,h_offset:h_offset+h_crop,w_offset:w_offset+w_crop,:]
    return im_R, im_T

def crop_augmentation_list(img_list):
    #Crop
    h_orig,w_orig = img_list[0].shape[1:3]
    h_crop = h_orig * 3 // 4 // 32 * 32
    w_crop = w_orig * 3 // 4 // 32 * 32
    
    try:
        w_offset = np.random.randint(0, w_orig-w_crop)
        h_offset = np.random.randint(0, h_orig-h_crop)
    except:
        print("Original W %d, desired W %d"%(w_orig,w_crop))
        print("Original H %d, desired H %d"%(h_orig,h_crop))
    crop_list = [img[:,h_offset:h_offset+h_crop,w_offset:w_offset+w_crop,:] for img in img_list]
    return crop_list
           


def tf_calculate_ADoLP(I_all):
    I1, I2, I3, I4 = I_all[:,:,:,:1], I_all[:,:,:,1:2], I_all[:,:,:,2:3], I_all[:,:,:,3:4]
    I = 0.5 * (I1 + I2 + I3 + I4)+1e-4
    Q = I1 - I3 
    U = I2 - I4
    zero_mat = tf.zeros(tf.shape(I1), tf.float32)
    ones_mat = 1e-4 * tf.ones(tf.shape(I1), tf.float32)
    Q = tf.where(tf.equal(Q, zero_mat), ones_mat, Q)
    DoLP = tf.divide(tf.sqrt(tf.square(Q)+tf.square(U)), I)
    AoLP = 0.5*tf.atan(U/Q)
#    AoLP = (AoLP + 0.786)/(2*0.786)
    return AoLP, DoLP

def ADoLP_loss(gt, output):
    AoLP1, DoLP1 = tf_calculate_ADoLP(gt)
    AoLP2, DoLP2 = tf_calculate_ADoLP(output)	
    AoLP_loss = tf.reduce_mean(tf.abs(AoLP1 - AoLP2))
    DoLP_loss = tf.reduce_mean(tf.abs(DoLP1 - DoLP2))
    return AoLP_loss + DoLP_loss


def GC_augmentation(im_in):
    #Flip
    magic = np.random.random()
    # print(im_in.shape)

    if magic > 0.75:
        im_in=im_in[:,::-1,:,:]
    elif magic < 0.25:
        im_in=im_in[:,:,::-1,:]
    #Crop
    h_orig,w_orig = im_in.shape[1:3]
    h_crop = 224
    w_crop = 224
    
    try:
        w_offset = np.random.randint(0, w_orig-w_crop-1)
        h_offset = np.random.randint(0, h_orig-h_crop-1)
    except:
        print("Original W %d, desired W %d"%(w_orig,w_crop))
        print("Original H %d, desired H %d"%(h_orig,h_crop))
    im_in=im_in[:,h_offset:h_offset+h_crop,w_offset:w_offset+w_crop,:]

    return im_in
 
