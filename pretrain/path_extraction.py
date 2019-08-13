####taking good slides forom WSI
# from WSI_reading import reading_WSI
import os
# from sampling import random_croping
import numpy as np
import random
import pandas as pd
os.environ
from openslide import (OpenSlide, OpenSlideError,
                       OpenSlideUnsupportedFormatError)
import re
import sys
import PIL
import numpy as np
import os
from PIL import Image, ImageDraw
from openslide.deepzoom import DeepZoomGenerator as dz
import cv2
import math
import cv2
import os
import collections
import numpy as np
import random
import shutil


def stainremover_small_patch_remover(img):
    cleaned_Images = []

    # img = cv2.imread(inputimage)
    # img = cv2.imread(inputimage,0)
    if len(img) < 256 or len(img[0]) < 256:
        #         print("lessdemsion")
        return None
        # print(inputimage)
        # os.remove(inputimage)
    else:
        # print('here')
        Xb = []
        Xg = []
        Xr = []
        for i in range(len(img)):
            Xb.append(np.mean(img[i][:, 0]))
            Xg.append(np.mean(img[i][:, 1]))
            Xr.append(np.mean(img[i][:, 2]))
        #     print(inputimage)
        #     print(np.mean(Xr),np.mean(Xg),np.mean(Xb))

        if np.mean(Xr) < 70 or np.mean(Xr) > 220:
            # print(np.mean(Xr))
            # print("red")
            #             print(inputimage.split('/')[-1])
            return None
            # print(inputimage)
            # os.remove(inputimage)

        elif np.mean(Xg) < 80 or np.mean(Xg) > 200:
            # print("green")
            return None
            # print(inputimage)
            # os.remove(inputimage)

        elif np.mean(Xb) < 100 or np.mean(Xb) > 210:
            # print("blue")
            return None
        else:
            # print('here')

            cv2.imwrite("temp0.png", img)
            img_bg = cv2.imread("temp0.png", 0)
            # img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if (img_bg.mean() < 215) and (img_bg.mean() > 50):

                return img

            else:
                # print((img_bg.mean()))

                # print("Blackandwhite")
                return None


def reading_WSI(slide):
    """
    Input: mask  -- a resulting mask containing just good nuclei, and all the others regions are white
           slide -- complete region of interest from the original slide
           values -- array of point annotations for each patch

    output: return an array of resultant matrices for each slide and its corresponding mask
    """
    x_start = 230
    """Zoom_level_20X"""
    width = slide.level_dimensions[0][0]  # 10666
    height = slide.level_dimensions[0][1]
    x_end = x_start + width
    y_start = 0
    # chunk_width = int(math.floor(width / 3))
    # chunk_height = int(math.floor(height / 3))
    # stratify each region of the slide into four equal parts
    print(slide.level_dimensions)
    slide1 = slide.read_region((x_start, y_start), 1,
                               (slide.level_dimensions[0][0], slide.level_dimensions[0][1])).convert('RGB')
    print("here")
    # slide1.save("slide_roi1.png")
    slide1 = np.asarray(slide1, dtype="int32")

    # slide1 = cv2.imread("slide_roi1.png")
    # print("Slide1 done")
    # slide2 = slide.read_region(((x_start + chunk_width), y_start), 0, (chunk_width, chunk_height)).convert('RGB')
    # slide2.save("slide_roi2.png")
    # slide2 = cv2.imread("slide_roi2.png")
    # print("Slide2 Done")
    # slide3 = slide.read_region(((x_start + chunk_width + chunk_width), y_start), 0,
    #                            (chunk_width, chunk_height)).convert('RGB')
    # slide3.save("slide_roi3.png")
    # slide3 = cv2.imread("slide_roi3.png")
    # print("Slide3 done")
    # slide4 = slide.read_region((x_start, chunk_height), 0, (chunk_width, chunk_height)).convert('RGB')
    # slide4.save("slide_roi4.png")
    # slide4 = cv2.imread("slide_roi4.png")
    #
    # slide5 = slide.read_region(((x_start + chunk_width), chunk_height), 0, (chunk_width, chunk_height)).convert('RGB')
    # slide5.save("slide_roi5.png")
    # slide5 = cv2.imread("slide_roi5.png")
    #
    # slide6 = slide.read_region(((x_start + chunk_width + chunk_width), chunk_height), 0,
    #                            (chunk_width, chunk_height)).convert('RGB')
    # slide6.save("slide_roi6.png")
    # slide6 = cv2.imread("slide_roi6.png")
    #
    # slide7 = slide.read_region((x_start, chunk_height + chunk_height), 0, (chunk_width, chunk_height)).convert('RGB')
    # slide7.save("slide_roi7.png")
    # slide7 = cv2.imread("slide_roi7.png")
    #
    # slide8 = slide.read_region((x_start + chunk_width, chunk_height + chunk_height), 0,
    #                            (chunk_width, chunk_height)).convert('RGB')
    # slide8.save("slide_roi8.png")
    # slide8 = cv2.imread("slide_roi8.png")
    #
    # slide9 = slide.read_region(((x_start + chunk_width + chunk_width), chunk_height + chunk_height), 0,
    #                            (chunk_width, chunk_height)).convert('RGB')
    # slide9.save("slide_roi9.png")
    # slide9 = cv2.imread("slide_roi9.png")
    #
    # print("Slide spliting into 9 done")
    # return [slide1, slide2, slide3, slide4, slide5, slide6, slide7, slide8, slide9]
    return slide1


def random_croping(img, tile_size):
    img_shape = img.shape
    # tile_size = (256, 256)
    x_point = random.randint(0, img_shape[0])
    y_point = random.randint(0, img_shape[1])
    try:
        # if (cropped_img.mean() > 50) and (cropped_img.mean() < 180) and (len(list(set(cropped_img.flatten()))) > 50):
        cropped_image = img[y_point:y_point + 256, x_point:x_point + 256]

        cleaned_image = stainremover_small_patch_remover(cropped_image)

    except:
        cleaned_image = None
    return cleaned_image


def path_extraction(img, path):
    tile_size = (256, 256)

    len_list = []
    for i in range(100000):

        patchs = random_croping(img, tile_size)
        if len(len_list) < 2000:
            if patchs != None:
                cv2.imwrite('%s/%s.png' % (path, i), patchs)
                len_list.append(i)

        else:
            return
    return
def total_data():
    Sample_data = pd.read_csv('match_sample_survival_info.csv')
    Sample_data_list = Sample_data['SAMPLE_ID'].values
    return Sample_data_list

if __name__ == "__main__":
    svs = []
    for filename in os.listdir("./GBM_TOP/"):
        svs.append(filename)

    slide_counter = 0
    for num in range(0, len(svs)):  # len(svs)):

        temp_svs = "./GBM_TOP/" + str(svs[num])  # patches collected already from 1004347, 1004366, 1004346"
        # print(temp_xm

        print(temp_svs)
        Sample_list = total_data()
        if str(svs[num][:15]) in Sample_list:
            path = "./GBM_top_patches2/Pretrain11/" + str(svs[num][:15])
            print(path)

            try:
                os.mkdir(path)
                print(num)
                slide = OpenSlide(temp_svs)
                patches = reading_WSI(slide=slide)
                path_extraction(patches, path)
            except:
                print('already_exist')
                # print(a)
        else:
            print("%s:::doesnotexist"%str(svs[num]))


