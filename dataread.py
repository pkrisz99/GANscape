# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:21:10 2020

@author: KR
"""

import os
import os
from PIL import Image
import numpy as np
import random
import csv


class DataRead:
    def __init__(self, folder_path, data_type, batch_size, min_images):
        self.folder_path = folder_path
        self.type = data_type
        if (batch_size*((int(min_images / batch_size))) < min_images):
            self.length = batch_size*((int(min_images / batch_size))+1)
        else:
            self.length = min_images
        self.batch_size = batch_size
        self.batch_num = int(self.length / batch_size)
        
        self.end_point = self.length
        
        if (self.type == 'train'):
            self.target_path = 'places_train\\'
            self.cropped_path = 'places_train_cropped'
            self.crop_path = 'places_train_crop'
        elif (self.type == 'valid'):
            self.target_path = 'places_valid\\'
            self.cropped_path = 'places_valid_cropped'
            self.crop_path = 'places_valid_crop'
        elif (self.type == 'test'):
            self.target_path = 'places_test\\'
            self.cropped_path = 'places_test_cropped'
            self.crop_path = 'places_test_crop'
        else:
            print("error, no such data_type")
            return
        # ======================target images=====================
        path = os.path.join(self.folder_path, self.target_path)
        
        images = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.jpg'): images.append(os.path.join(root, file))
        print(len(images))
                
        tmp = np.zeros((self.batch_num, self.batch_size, 256, 256, 3))
        for i in range(self.batch_num):
            for j in range(self.batch_size):
                img = Image.open(images[i*batch_size + j])
                img_temp=np.array(img)
                tmp[i,j]  =  img_temp
                    
        self.target_images = tmp
        # ============================train images-cropped=============================
        path = os.path.join(self.folder_path, self.cropped_path)
        
        images = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.jpg'): images.append(os.path.join(root, file))
        print(len(images))
                
        tmp = np.zeros((self.batch_num, self.batch_size, 256, 256, 3))
        for i in range(self.batch_num):
            for j in range(self.batch_size):
                img = Image.open(images[i*batch_size + j])
                img_temp=np.array(img)
                tmp[i,j]  =  img_temp
                    
        self.cropped_images = tmp
        # =============================train images-crop===========================
        path = os.path.join(self.folder_path, self.crop_path)
        
        images = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.jpg'): images.append(os.path.join(root, file))
        print(len(images))    
                
        tmp_big = []
        for i in range(self.batch_num):
            tmp_small = []
            for j in range(self.batch_size):
                img = Image.open(images[i*batch_size + j])
                img_temp=np.array(img)
                tmp_small.append(img_temp)
            
            tmp_big.append(tmp_small)
                    
        self.crop_images = tmp_big
        # =====================================result images===========================
        # preparing the space
        self.result_images = np.zeros((self.batch_num, self.batch_size, 256, 256, 3))
        #===================================reading in the csv
        csv_path = 'out_' + self.type + '.csv'
        with open(csv_path, "r") as infile:
            r = csv.reader(infile)
            tmp_big = []
            for i in range(self.batch_num):
                tmp_small = []
                for j in range(self.batch_size):
                    row = next(r)
                    row = row[1:]
                    row = list(map(int, row)) #itt lehetne még egy np.array alakítás
                    tmp_small.append(row)
                
                tmp_big.append(tmp_small)
        
        self.csv = tmp_big
                 
        #important parts
        # self.batch_num
        # self.batch_size
        # self.target_images
        # self.cropped_images
        # self.crop_images
        # self.result_images
        # self.csv
        
            
    
    
    
    
    
    
    
    
    
        