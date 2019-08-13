from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout,concatenate,Input,Conv1D, MaxPooling1D,Flatten, Conv2DTranspose,SpatialDropout2D,Dropout

import pandas as pd
from keras.models import Model
import numpy as np
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import accuracy_score
from keras import optimizers
import os
import random
from sklearn.model_selection import train_test_split,KFold
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
#random.seed(42)
import os
from os import walk
# from Datageneator import data_generator
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.callbacks import EarlyStopping, ModelCheckpoint

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
def input(inputshape):
    input_img = Input(shape = (256,256,1))
    return input_img

def Aspp(inputmodel):

    model1 = Conv2D(50, kernel_size=(5,5),dilation_rate=2,activation='relu')(inputmodel)
    #model2 = Conv2D(50, kernel_size=(3,3),dilation_rate=4,activation='relu')(model1)
    #model3 = Conv2D(50, kernel_size=(5,5),dilation_rate=2,activation='relu')(model2)
    #model4 = Conv2D(50, kernel_size=(5,5),dilation_rate=2,activation='relu')(model3)
    #outputmodel =     concatenate([model1, model2,model3,model4],axis = 0)
    return model1



def layer1(inputmodel):
    #model = Model()(inputmodel)
    model = Conv2D(50, kernel_size=(5,5),dilation_rate=2,activation='relu')(inputmodel)
    model = SpatialDropout2D(0.3)(model)
    model = Conv2D(50, kernel_size=(5, 5), dilation_rate=2, activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    # model = SpatialDropout2D(0.3)(model)
    model = Conv2D(50, kernel_size=(5,5),dilation_rate=2,activation='relu')(model)
    model = SpatialDropout2D(0.3)(model)
    model = Conv2D(50, kernel_size=(5, 5), dilation_rate=2, activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    # model = SpatialDropout2D(0.3)(model)
    model = Conv2D(50, kernel_size=(5,5),dilation_rate=2,activation='relu')(model)
    model = SpatialDropout2D(0.3)(model)
    model = Conv2D(50, kernel_size=(5, 5), dilation_rate=2, activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)

    model = SpatialDropout2D(0.3)(model)
    return model
def layerfinalcnn(inputmodel):
   # modelf = Conv2D(50, kernel_size=(5, 5), activation='relu')(inputmodel)
    #modelf = Conv2DTranspose(50, kernel_size=(3, 3), activation='relu')(inputmodel)
    return modelf




def model(inputshape):
    input_img = input(inputshape)
    outputmodel = layer1(input_img)
    #outputmodel = Aspp(outputmodel)
    #outputmodel = Aspp(outputmodel)
    #outputmodel = layerfinalcnn(outputmodel)

    outmodelf = Flatten()(outputmodel)
    model = Dense(units=30, activation='relu', input_dim=50)(outmodelf)
    model = Dropout(0.5)(model)
    # model = Dense(units=30, activation='relu', input_dim=50)(outmodelf)
    model = Dense(units=1, activation='linear')(model)
    fmodel = Model(input_img, model)
    return fmodel

### DATA LOADER

datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.0,
        zoom_range=0.0,
        horizontal_flip=True)

train_generator = datagen.flow_from_directory(
    directory=r"./PSB_pretrain_1/Train/",
    target_size=(256, 256),
    color_mode="grayscale",
    batch_size=32,
    class_mode='binary',
    shuffle=True,
    seed=42
)






valid_generator = datagen.flow_from_directory(
    directory=r"./PSB_pretrain_1/Validation/",
    target_size=(256, 256),
    color_mode="grayscale",
    batch_size=32,
    class_mode='binary',
    shuffle=True,
    seed=42
)



test_generator = datagen.flow_from_directory(
    directory=r"./PSB_pretrain_1/Test/",
    target_size=(256, 256),
    color_mode="grayscale",
    batch_size=1,
    class_mode='binary',
    shuffle=False,
    seed=42
)

batch_size = 32
# num_classes = 4


##main
# X_train, X_test, y_train, y_test = data_generator("GBM_top_patches","match_sample_survival_info.csv",0.8,1)
# X_train = np.array(X_train)
# y_train = np.array(y_train)
# X_train = np.array(X_test)
# y_train = np.array(y_train)
# X1_test = np.expand_dims(X_test, axis=3)
# X1_train = np.expand_dims(X_test, axis=3)
# SGD = optimizers.SGD(lr=0.00001, decay=1e-7, momentum=0.9, nesterov=True)
adam = optimizers.adam(lr=0.00001,decay=1e-7)
fmodel =  model((256,256))
fmodel.compile(optimizer = adam, loss = 'mean_squared_error')

#fmodel.fit(X1_train, y_train, validation_split=0.15, batch_size=256, epochs=100)

train_generator.n//train_generator.batch_size


#steps =STEP_SIZE_VALID
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
# callbacks = [EarlyStopping(monitor='val_loss', patience=5),
#              ModelCheckpoint(filepath='best_model5.h5', monitor='val_loss', save_best_only=True)]
results = fmodel.fit_generator(generator=train_generator,


                               steps_per_epoch=STEP_SIZE_TRAIN,
                                        validation_data=valid_generator,
validation_steps=STEP_SIZE_VALID,
                               epochs = 30


)
fmodel.save('pretrain_pagenet.h5')
# model.evaluate_generator(generator=valid_generator,steps =STEP_SIZE_VALID )
y_pred = fmodel.predict_generator(test_generator,steps = test_generator.n, verbose=1)
labelpredict = []
labelpredict = []
labeltest = []
labelpredict.extend(y_pred)
# labeltest.extend(y_test)
labelpredictdf = pd.DataFrame(labelpredict)
labelpredictdf.to_csv("labelpredict10.csv")
# labeltestdf = pd.DataFrame(labeltest)
# labeltestdf.to_csv("labeltest.csv")












