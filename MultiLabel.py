import numpy as np
import pandas as pd
import csv
import cv2

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.dataset import Dataset
import os.path
from os import path
from itertools import cycle, islice

#bag of words of labels
def gen_set(csvfile, outputfile):
    disease_list = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', \
                   'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', \
                   'Hernia']
    alldiseases = {disease:i for i,disease in enumerate(disease_list)}
    with open(outputfile, 'w') as fp:
        with open(csvfile, 'r') as cfile:
            line = csv.reader(cfile, delimiter=',')
            index = 0
            for element in line:
                # the first column is the image filename, while the second
                # column has the list of diseases separated by |
                diseases = element[1].split('|')
                #fp.write('%d\t'%index)
                for d in alldiseases:
                    if ((d in diseases) and (d == 'Atelectasis')):
                        fp.write('%d\t'%1)
                    elif((d in diseases) and (d == 'Consolidation')):
                        fp.write('%d\t'%2)
                    elif((d in diseases) and (d == 'Infiltration')):
                        fp.write('%d\t'%3)
                    elif((d in diseases) and (d == 'Pneumothorax')):
                        fp.write('%d\t'%4)
                    elif((d in diseases) and (d == 'Edema')):
                        fp.write('%d\t'%5)
                    elif((d in diseases) and (d == 'Emphysema')):
                        fp.write('%d\t'%6)
                    elif((d in diseases) and (d == 'Fibrosis')):
                        fp.write('%d\t'%7)
                    elif((d in diseases) and (d == 'Effusion')):
                        fp.write('%d\t'%8)
                    elif((d in diseases) and (d == 'Pneumonia')):
                        fp.write('%d\t'%9)
                    elif((d in diseases) and (d == 'Pleural_Thickening')):
                        fp.write('%d\t'%10)
                    elif((d in diseases) and (d == 'Cardiomegaly')):
                        fp.write('%d\t'%11)
                    elif((d in diseases) and (d == 'Nodule')):
                        fp.write('%d\t'%12)
                    elif((d in diseases) and (d == 'Mass')):
                        fp.write('%d\t'%13)
                    elif((d in diseases) and (d == 'Hernia')):
                        fp.write('%d\t'%14)
                    else:
                        fp.write('%d\t'%0)
                fp.write('images/%s\n' % element[0])
                index += 1

path = 'D:/EIT_AUS_TUB/SoSe2020_MLInMIP/MedicalImageProcessing/'    
os.chdir(path)
gen_set('traindata.csv', 'chestxraytrain.txt')
gen_set('valdata.csv', 'chestxrayval.txt')
gen_set('testdata.csv','chestxraytest.txt') 

def Diagnosis(word):
    #print("word", word)
    if(word == 1):
        diag = 'Atelectasis'
    elif(word == 2):
        diag = 'Consolidation'
    elif(word == 3):
        diag = 'Infiltration'
    elif(word == 4):
        diag = 'Pneumothorax'
    elif(word == 5):
        diag = 'Edema'
    elif(word == 6):
        diag = 'Emphysema'
    elif(word == 7):
        diag = 'Fibrosis'
    elif(word == 8):
        diag = 'Effusion'
    elif(word == 9):
        diag = 'Pneumonia'
    elif(word == 10):
        diag = 'Pleural_Thickening'
    elif(word == 11):
        diag = 'Cardiomegaly'
    elif(word == 12):
        diag = 'Nodule'
    elif(word == 13):
        diag = 'Mass'
    elif(word == 14):
        diag = 'Hernia'
    else:
        diag = "Undiagnosed"

    #print(diag)
    return diag


class XRaysTrainDataset(Dataset):
    
    def __init__(self, csv_name, transform=None):
        file1 = open(csv_name, "r")
        self.data =  file1.readlines()           
        self.data_len = len(self.data)         
        self.transform = transform
             
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):       
        #read labels in each line in the txt file
        cnt = 0 
        imageName = []            
        imgLab = []
        for word in self.data[index].split():          
            #diagnose until the last index which is the image name
            if(cnt < 14):
                #print(word)
                diag = Diagnosis(int(word))
                #print(diag)
                if(diag != 'Undiagnosed'):
                    imgLab.append(diag)

            if(cnt == 14):
                imageName.append(word)
            cnt+=1 
         
        if not imgLab:
            imgLab.append("Undiagnosed")

        return imageName, imgLab           
    

# Define transforms
train_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(), # randomly flip and rotate
                                        transforms.RandomRotation(10),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


traindataLoader = XRaysTrainDataset('chestxraytrain.txt', transform = train_transform)
trainLoader = torch.utils.data.DataLoader(traindataLoader, batch_size = 1, shuffle = True)

batch_size = 15
print("in trainloader")
for b in range(batch_size):
    dataiter = iter(trainLoader)
    images, labels = dataiter.next()
    print(images)
    print(labels)
    print("--------------------------------------------------------------------")




