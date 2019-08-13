#creator : sai kosaraju
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import random
import cv2
import shutil
from os import walk
from shutil import copyfile
def reading_label_csv(inputfile):
    dataframe = pd.read_csv(inputfile)
    patientslist = dataframe.values.tolist()
    return patientslist
def gettingfilenames(path):
    f = []
    for (dirpath, dirnames, filenames) in walk(path):
        f.extend(filenames)
    return f


def getting_nonsensor_data(patientslist, sensor_value):
    # patientslist = reading_label_csv("C:/Users/skosara1/Kennesaw State University/Jie Hao - Genomic Data","match_sample_survival_info.csv")
    decessed_patients = []

    for i in range(len(patientslist)):
        if patientslist[i][2] == sensor_value:
            decessed_patients.append((patientslist[i][0], patientslist[i][1]))
    return decessed_patients


def reading_folders_of_patches(inputfolderofpatches):
    imagepatientlist = os.listdir(inputfolderofpatches)

    return imagepatientlist


def patientslist_with_labels(decessed_patients, imagepatientlist):
    patientlist = []
    for i in range(len(decessed_patients)):

        for j in range(len(imagepatientlist)):
            if decessed_patients[i][0] == imagepatientlist[j][:15]:
                patientlist.append((imagepatientlist[j], decessed_patients[i][1]))

    return patientlist
def train_test_split_random(finalpatients):
    random.shuffle(finalpatients)
    X= finalpatients
    # X1 = finalpatients[:]
    # Y = finalpatients[int(len(finalpatients)*perx):]
    return X


def train_data_generator(X, inputfolder):
    X_train = []
    y_train = []

    for i in range(len(X)):
        files = gettingfilenames("%s/%s" % (inputfolder, X[i][0]))
       # print(i, X[i][1])

        for j in range(len(files)):
            img_x = cv2.imread("%s/%s/%s" % (inputfolder, X[i][0], files[j]), 0)
            X_train.append(img_x)
            # print(img_x)
            y_train.append(X[i][1])
        # print(list(set(y_train1)))
    #         X_train.extend(X_train1)
    #         y_train.extend(y_train1)

    return X_train, y_train


def test_data_generator(y, inputfolder):
    X_test = []
    y_test = []

    for i in range(len(y)):
        files = gettingfilenames("%s/%s" % (inputfolder, y[i][0]))

        for j in range(len(files)):
            img_x = cv2.imread("%s/%s/%s" % (inputfolder, y[i][0], files[j]), 0)
            X_test.append(img_x)
            # print(X[i][1])
            y_test.append(y[i][1])
            #print(y[i][1])

    return X_test, y_test


def data_generator(inputfolder, csvfile, sensorvalue):
    patientslist = reading_label_csv(csvfile)
    # getting_nonsensor_data(patientslist,sensor_value)
    decessed_patients = getting_nonsensor_data(patientslist, sensorvalue)
    imagepatientlist = reading_folders_of_patches(inputfolder)
    finalpatientslist = patientslist_with_labels(decessed_patients, imagepatientlist)
    X, y = train_test_split_random(finalpatientslist)
    #print(X, y)
    X_train, y_train = train_data_generator(X, inputfolder)
    random.seed(42)
    random.shuffle(X_train)
    random.seed(42)
    random.shuffle(y_train)

    X_test, y_test = test_data_generator(y, inputfolder)
    random.seed(42)
    random.shuffle(X_test)
    random.seed(42)
    random.shuffle(y_test)

    return X_train, X_test, y_train, y_test
def datamover(inputfolder,csvfile,sensorvalue):
    patientslist = reading_label_csv(csvfile)
    #getting_nonsensor_data(patientslist,sensor_value)
    decessed_patients= getting_nonsensor_data(patientslist,sensorvalue)
    imagepatientlist = reading_folders_of_patches(inputfolder)
    finalpatientslist = patientslist_with_labels(decessed_patients,imagepatientlist)
    X = train_test_split_random(finalpatientslist)

    return X
def sampling_with_replacment(list_dir,n):
    list_dir_new = []
    for i in range(n):
        x = random.sample(list_dir,1)
        list_dir_new.extend(x)
    return list_dir_new

def resampling(inputfolder1,n):
    if len(os.listdir(inputfolder1)) > n:
        list_dir_sampled = random.sample(os.listdir(inputfolder1),n)
        list_dir_n = []
        for p in range(len(list_dir_sampled)):
            list_dir_n.append((list_dir_sampled[p],'%s_%s.png'%(list_dir_sampled[p][:-4],p)))

    elif len(os.listdir(inputfolder1)) > 0:
        # print(len(os.listdir(inputfolder1)))
        list_dir = os.listdir(inputfolder1)
        list_dir_sampled = sampling_with_replacment(list_dir,n)
        # except:
        #     print(inputfolder1)
        #     print("is empty")
        #     print(a)
        #     list_dir_sampled = []
    # print(len(list_dir_sampled))
    # list_dir_n = []

        list_dir_n = []
        for p in range(len(list_dir_sampled)):
            list_dir_n.append((list_dir_sampled[p],'%s_%s.png'%(list_dir_sampled[p][:-4],p)))
    else:
        print('empty')
        list_dir_n =[]
        # return None
    return list_dir_n

def moving(inputfolder,outputfolder,data_single,data_label,n):
    os.mkdir("%s/%s" % (outputfolder, data_label))
    list_moving = resampling("%s/%s"%(inputfolder,data_single),n)
    for i in range(len(list_moving)):

        shutil.copy2("%s/%s/%s"%(inputfolder,data_single,list_moving[i][0]),"%s/%s/%s" % (outputfolder, data_label,list_moving[i][1]))
    print(len(os.listdir("%s/%s" % (outputfolder, data_label))))
    return

### change isin['Valid] as train,test,valid and run the code three times
def renamer(inputfolder,outputfolder,data,n):
    split_index = pd.read_csv("/home/sai/example_split.csv")
    train_index = split_index[split_index.Split.isin(['Valid'])].SAMPLE_ID.values

    for i in range(1,len(data)):
        if data[i][0][:15] in train_index:
            #print(data[i][0])
            try:
                # os.mkdir("%s/%s"%(outputfolder,data[i][1]))

                # shutil.move("%s/%s"%(inputfolder,data[i][0]),"%s/%s"%(outputfolder,data[i][1]))
                moving(inputfolder, outputfolder, data[i][0], data[i][1], n)
            except:

                print("%s/%s"%(inputfolder,data[i][0]))
                # print (a)

    return
##change directory inputfolder,outputfolder
def main():
    X= datamover("/home/sai/GBM_top_patches2/Pretrain",
                         "/home/sai/match_sample_survival_info.csv", 1)
    #print(X)


    renamer("/home/sai/GBM_top_patches2/Pretrain","/home/sai/PSB_pretrain_1/Valid",X,1000)
    # renamer("/home/SharedStorage2/nelson/cancer_detection_project/GBM_top_patches", "/home/SharedStorage3/sai/expirement/Test", y)
    # renamer("/home/SharedStorage2/nelson/cancer_detection_project/GBM_top_patches", "/home/SharedStorage3/sai/expirement/Validation", X1)
    return
if __name__ == "__main__":
    main()
    # X, X1, y = datamover("/home/SharedStorage2/nelson/cancer_detection_project/GBM_top_patches","/home/SharedStorage2/nelson/cancer_detection_project/match_sample_survival_info.csv",0.8,1)
    # renamer("/home/SharedStorage2/nelson/cancer_detection_project/GBM_top_patches","expirement/Train",X)
    # renamer("/home/SharedStorage2/nelson/cancer_detection_project/GBM_top_patches", "expirement/Test", y)
    # renamer("/home/SharedStorage2/nelson/cancer_detection_project/GBM_top_patches", "expirement/Validation", X1)
