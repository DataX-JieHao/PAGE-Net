
import numpy as np
import pandas as pd
from keras.models import load_model
from keract import get_activations
import cv2
import os
from random import shuffle
import keract

import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# from matplotlib import pyplot
##inter pooling

def normfunction(featuremap, k, normvalue):
    featuremap_sorted = sorted(featuremap, reverse=True)
    featuremap_pooled = featuremap_sorted[:k]
    featuremap_pooled = np.array(featuremap_pooled)

    featuremap_score = ((float(sum((featuremap_pooled) ** normvalue) / k)) ** (1. / normvalue)).real

    return featuremap_score


def feature_map_scoring(featuremap, k1, normvalue):
    featuremap = np.array(featuremap)

    featuremap_flatten = featuremap.flatten()
    #     print(featuremap_flatten)
    featuremap_score = normfunction(featuremap_flatten, k1, normvalue)
    #     print(featuremap_score)
    return featuremap_score


def feature_map_pooling(featuremaps, k1, normvalue):
    intra_patch_pooling = []
    for i in range(len(featuremaps[0][0][0])):
        #         print(len(featuremaps[0][0][0]))
        featuremap_score = feature_map_scoring(featuremaps[:, :, :, i], k1, normvalue)
        # print(featuremap_score)
        intra_patch_pooling.append(featuremap_score)
    # print(intra_patch_pooling)
    return intra_patch_pooling


def reading_model(input_model):
    model = load_model(input_model)
    return model


def extracting_feature_maps(model, inputimage, layer_number):
    img_n = cv2.imread(inputimage, 0)
    img = np.expand_dims(img_n, axis=2)
    img = np.expand_dims(img, axis=0)

    feature_maps = get_activations(model, img)

    key_dict = []
    key_dict.extend(feature_maps.keys())
    #     keract.display_activations(feature_maps)
    featuremaps = feature_maps[key_dict[layer_number]]

    return featuremaps


def patch_pooling(pooled_patchs, k2, normvalue):
    pooled_patchs = np.array(pooled_patchs)
    patch_pooling = []
    for i in range(len(pooled_patchs[0])):
        #         print(len(pooled_patchs[0]))
        if len(pooled_patchs[0]) < k2:
            pooled_patch_column = pooled_patchs[:, i]
            pooled_patch_score = normfunction(pooled_patch_column, len(pooled_patchs[0]), normvalue)
            patch_pooling.append(pooled_patch_score)
        else:
            pooled_patch_column = pooled_patchs[:, i]
            pooled_patch_score = normfunction(pooled_patch_column, k2, normvalue)
            patch_pooling.append(pooled_patch_score)
    return patch_pooling


def image_data(patient_id, N, k1, k2, model):
    '''Input
    N: Number of patches
    '''

    total_patches = os.listdir(patient_id)
    # N = len(patient_id)

    shuffle(total_patches)

    pooled_patchs = []
    input_image = []
    for i in range(N):
        try:

            # print(str("%s/%s"%(patient_id,total_patches[i])))
            inputimage = str("%s/%s" % (patient_id, total_patches[i]))
            # plt.imshow(cv2.imread(inputimage))
            featuremaps = extracting_feature_maps(model, inputimage, 12)
            # print(featuremaps)
            feature_map_score = feature_map_pooling(featuremaps, k1, 3)
            # print(np.argmax(feature_map_score))
            pooled_patchs.append(feature_map_score)
            input_image.append("%s/%s" % (patient_id, total_patches[i]))
        except:
            print(inputimage)
    #             print(e)
    image_pooling = patch_pooling(pooled_patchs, k2, 3)
    important_feature_map = np.argmax(image_pooling)
    # return image_pooling,pooled_patchs,important_feature_map,input_image
    #     print(image_pooling[1])
    return image_pooling


if __name__ == '__main__':
    # image_pooling,pooled_patchs,important_feature_map,input_image = image_data('C:/Users/skosara1/OneDrive - Kennesaw State University/ACMBCB/Split8/TCGA-41-4097-01A-01-TS1',700,387,15)

    #     split_index = pd.read_csv("BCB_SPLIT_fold1_newsplit1.csv")
    #     train_index = split_index[split_index.Data_usage_type.isin(['Validation'])].Patient_ID.values
    # print(train_index)
    model = reading_model("pretrain_pagenet17.h5")
    image_train_id = os.listdir(r"/home/sai/GBM_top_patches2/Pretrain8")
    train_id = []
    train_index_t = []
    x = len(image_train_id)
    print(len(image_train_id))

    for g in range(0,x):

        # len(image_train_id)

        #         if image_train_id[g][:15] in train_index:
        try:

            print(g)
            print(image_train_id[g])
            if len(os.listdir(r"/home/sai/GBM_top_patches2/Pretrain8/%s" % image_train_id[g])) > 1000:
                print(len(os.listdir(r"/home/sai/GBM_top_patches2/Pretrain8/%s" % image_train_id[g])))

                train_ID = image_data(r"/home/sai/GBM_top_patches2/Pretrain8/%s" % image_train_id[g], 1000, 65,
                                      100, model)
                print("large")
            else:
                print(len(os.listdir(r"/home/sai/GBM_top_patches2/Pretrain8/%s" % image_train_id[g])))
                train_ID = image_data(r"/home/sai/GBM_top_patches2/Pretrain8/%s" % image_train_id[g], len(
                    os.listdir(r"/home/sai/GBM_top_patches2/Pretrain8/%s" % image_train_id[g])), 65, 100, model)

                print("small")
        except:
            train_ID = []
            #             print(w)
            print("Split7/%s" % image_train_id[g])
            for i in range(50):
                train_ID.append(0)

        print(train_ID)
        train_ID.append(image_train_id[g])
        train_index_T = image_train_id[g]
        train_index_t.append(train_index_T)

        train_id.append(train_ID)
    train_index_tdf = pd.DataFrame(train_index_t)
    train_index_tdf.to_csv('Aggregation.csv')
    train_iddf = pd.DataFrame(train_id)
    train_iddf.to_csv('Aggregationindext.csv')


