import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
import rawpy
import os
import exifread


def demosaic(raw_array, exif_dict=None):
    if str(exif_dict["Image Make"]) == "HUAWEI":
        rgb = np.stack([raw_array[1::2, 1::2], (raw_array[0::2, 1::2] + raw_array[1::2, 0::2]) / 2,
                    raw_array[0::2, 0::2]], axis=2)
    else:
        rgb = np.stack([raw_array[0::2, 0::2], (raw_array[0::2, 1::2] + raw_array[1::2, 0::2]) / 2,
                        raw_array[1::2, 1::2]], axis=2)
    neutral_wb =tag2matrix(str(exif_dict["EXIF:AsShotNeutral"]))     
    rgb = np.concatenate([rgb[...,0:1].clip(0,neutral_wb[0,0]), 
                                   rgb[...,1:2].clip(0,neutral_wb[1,0]), 
                                   rgb[...,2:3].clip(0,neutral_wb[2,0])], axis=2)

    return rgb#.clip(0,1.0)

def Linearization(raw_bayer, exif_dict):
    black_level = exif_dict["BlackLevel"]
    white_level = exif_dict["WhiteLevel"]
    raw_bayer = raw_bayer.astype(np.float32)
    raw_bayer = (raw_bayer - black_level) / (white_level - black_level)
    raw_linear = raw_bayer
    return raw_linear

def gamma_correction(rgb, gamma=2.2):
    return np.power(rgb, 1 / gamma)

def get_matrix(m1, m2, tp1, tp2, tp):
    if (tp < tp1):
        m = m1
    elif (tp > tp2):
        m = m2
    else:
        g = (1/ float(tp) - 1 / float(tp2)) / (1 / float(tp1) - 1 / float(tp2))
        m = g * m1 + (1-g) * m2
    return m

def WhiteBalance_ColorCalibration(rgb_demosaic, exif_dict):
    d50tosrgb = np.reshape(np.array([3.1338561, -1.6168667, -0.4906146, 
                                    -0.9787684, 1.9161415, 0.0334540, 
                                     0.0719453, -0.2289914, 1.4052427]), (3,3))
    d50toprophotorgb = np.reshape(np.array([1.3459433, -0.2556075, -0.0511118, 
                                            -0.5445989, 1.5081673, 0.0205351,
                                            0, 0, 1.2118128]), (3,3))

    height, width, channels = rgb_demosaic.shape
    forward_matrix1 = tag2matrix(str(exif_dict["EXIF:ForwardMatrix1"]))
    forward_matrix2 = tag2matrix(str(exif_dict["EXIF:ForwardMatrix2"]))
    color_matrix1 = tag2matrix(str(exif_dict["EXIF:ColorMatrix1"]))
    color_matrix2 = tag2matrix(str(exif_dict["EXIF:ColorMatrix2"]))
    camera_calibration1 = tag2matrix(str(exif_dict["EXIF:CameraCalibration1"]))
    camera_calibration2 = tag2matrix(str(exif_dict["EXIF:CameraCalibration2"]))
    neutral_wb =tag2matrix(str(exif_dict["EXIF:AsShotNeutral"])) 
    
    analog_balance = np.diag(np.asarray([float(i) for i in exif_dict["EXIF:AnalogBalance"].values])) #np.diag([1, 1, 1])
    
    
    
    rgb_demosaic = np.concatenate([rgb_demosaic[...,0:1].clip(0,neutral_wb[0,0]), 
                                   rgb_demosaic[...,1:2].clip(0,neutral_wb[1,0]), 
                                   rgb_demosaic[...,2:3].clip(0,neutral_wb[2,0])], axis=2)

    # Standard light A
    temparature1 = 2850
    # D65
    temparature2 = 6500

    if (exif_dict["EXIF:Make"] == "NIKON CORPORATION"):
        image_temparatue = 4000#exif_dict["MakerNotes:ColorTemperatureAuto"]
    elif (exif_dict["EXIF:Make"] == "HUAWEI"):
        # hack for HUAWEI
        image_temparatue = 4000

    forward_matrix = get_matrix(forward_matrix1, forward_matrix2, temparature1, temparature2, image_temparatue)
    camera_calibration = get_matrix(camera_calibration1, camera_calibration2, temparature1, temparature2, image_temparatue)
    rgb_reshaped = np.reshape(np.transpose(rgb_demosaic, (2,0,1)),(3,-1))
    prophotorgbtod50 = np.linalg.inv(d50toprophotorgb)
    ref_neutral = np.matmul(np.linalg.inv(np.matmul(analog_balance,camera_calibration)), neutral_wb)
    d = np.linalg.inv(np.diag([ref_neutral[0,0], ref_neutral[1,0], ref_neutral[2,0]]))
    camera2d50 = np.matmul(np.matmul(forward_matrix, d),
                          np.linalg.inv(np.matmul(analog_balance, camera_calibration)))
    camera2srgb = np.matmul(d50tosrgb, camera2d50)
    rgb_srgb = np.matmul(camera2srgb, rgb_reshaped)
    orgshape_rgb_srgb = np.reshape(np.transpose(rgb_srgb, (1, 0)),(height, width,3))
    return orgshape_rgb_srgb.clip(0, 1)


    # Step 7: Applying the hue / saturation / value mapping
    # Step 9: apply color mapping - no for HUAWEI
    # Step 10:Tone curve
    # Fitting curve
    # If the tone curve is not found, we use a default tone curve
    # Step 11: Convert to srgb
    # Step 12: Gamma Correction

def tag2matrix(info):
    value_list = info[1:-1].split(", ")
    value_array = []
    for value in value_list:
        if "/" in value:
            den, num = value.split("/")
            value_array.append(float(den) / float(num))
        else:
            value_array.append(float(value))
    if len(value_list) == 9:
        value_mat =  np.array(value_array).reshape([3,3])
    else:
        value_mat =  np.array(value_array).reshape([3,1])
    return value_mat

def prepare_exifdict(raw_path):
    # Open image file for reading (binary mode)
    f = open(raw_path, 'rb')

    # Return Exif tags
    exif_dict = exifread.process_file(f, details=True)

    exif_dict["EXIF:Make"] = str(exif_dict["Image Make"])
    if str(exif_dict["Image Make"]) == "NIKON CORPORATION":
        exif_dict["BlackLevel"] = 1008.0
        exif_dict["WhiteLevel"] = 16384.0
    elif str(exif_dict["Image Make"]) == "HUAWEI":
        exif_dict["BlackLevel"] = 256.0
        exif_dict["WhiteLevel"] = 4096.0

    exif_dict["EXIF:ColorMatrix1"] = exif_dict["Image Tag 0xC621"]
    exif_dict["EXIF:ColorMatrix2"] = exif_dict["Image Tag 0xC622"]
    exif_dict["EXIF:CameraCalibration1"] = exif_dict["Image Tag 0xC623"]
    exif_dict["EXIF:CameraCalibration2"] = exif_dict["Image Tag 0xC624"]
    exif_dict["EXIF:ForwardMatrix1"] = exif_dict["Image Tag 0xC714"]
    exif_dict["EXIF:ForwardMatrix2"] = exif_dict["Image Tag 0xC715"]
    exif_dict["EXIF:AsShotNeutral"] = exif_dict["Image Tag 0xC628"]
    exif_dict["EXIF:AnalogBalance"] = exif_dict["EXIF:AsShotNeutral"]
    return exif_dict

def prepare_rawlinear(raw_path, norm = False):
    exif_dict = prepare_exifdict(raw_path)
    raw_bayer = rawpy.imread(raw_path).raw_image_visible.copy()
    raw_linear = Linearization(raw_bayer.copy(), exif_dict)
    if norm == True:
        raw_nonexposure = raw_linear.copy()
        raw_nonexposure[raw_nonexposure>0.75] = 0
        raw_linear = raw_linear * (1.0 / raw_nonexposure.max())
    rgb_demosaic = demosaic(raw_linear, exif_dict)
    return rgb_demosaic


def process_raw_from_raw_linear(rgb_demosaic, exif_dict):
    orgshape_rgb_srgb = WhiteBalance_ColorCalibration(rgb_demosaic, exif_dict)
    rgb_gamma = gamma_correction(orgshape_rgb_srgb)
    return rgb_gamma


def obtain_rgb_flashonly(path_A, path_B, raw_camera="Huawei", norm=False):    
    input_raw1 = path_A
    input_raw2 = path_B
    if raw_camera == "Huawei":
        raw_flash_demosaic = prepare_rawlinear(input_raw1)
        raw_ambient_demosaic = prepare_rawlinear(input_raw2)
        raw_pureflash = (raw_flash_demosaic - raw_ambient_demosaic).clip(0, 1)

        if norm == True:
            raw_nonexposure = raw_flash_demosaic.copy()
            raw_nonexposure[raw_nonexposure>0.75] = 0
            ratio = 1.0 / raw_nonexposure.max()
            raw_pureflash = raw_pureflash * ratio
            raw_flash_demosaic = raw_flash_demosaic * ratio
            raw_ambient_demosaic = raw_ambient_demosaic * ratio
        
            
        exif_dict = prepare_exifdict(input_raw2)
        rgb_A_minus_B = process_raw_from_raw_linear(raw_pureflash, exif_dict)
        rgb_A = process_raw_from_raw_linear(raw_flash_demosaic, exif_dict)
        rgb_B = process_raw_from_raw_linear(raw_ambient_demosaic, exif_dict)
    else:
        black_level = {"Huawei":256, "Nikon":1024}
        with rawpy.imread(input_raw1) as bayer1:
            rgb_A=bayer1.postprocess()
            bayer2 = rawpy.imread(input_raw2)
            rgb_B=bayer2.postprocess()
            bayer1.raw_image_visible[:] = bayer1.raw_image_visible[:] - bayer2.raw_image_visible[:] + black_level[raw_camera]
            rgb_A_minus_B = bayer1.postprocess()
    return rgb_A, rgb_B, rgb_A_minus_B
raw_ambient_path = "flash.dng"
raw_flash_path = "ambient.dng"
rgb_flash, rgb_ambient, rgb_flashonly = obtain_rgb_flashonly(raw_flash_path, raw_ambient_path)
cv2.imwrite("rgb_ambient.jpg", rgb_ambient[::10,::10,::-1] * 255.)
cv2.imwrite("rgb_flash.jpg", rgb_flash[::10,::10,::-1]* 255.)
cv2.imwrite("rgb_flashonly.jpg", rgb_flashonly[::10,::10,::-1]* 255.)