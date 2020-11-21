# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:21:10 2020

@author: KR
"""

import os
from PIL import Image
import numpy as np
import random
import csv


class DataRead:
    def __init__(self, folder_path, data_type, batch_size, batch_num, shuffle = False, seed = False):
        self.folder_path = folder_path
        self.type = data_type
        self.shuffle = shuffle
        self.seed = seed
       
        self.length = batch_num*batch_size
            
        self.batch_size = batch_size
        self.batch_num = batch_num
        
        self.end_point = self.length
        self.made_masks = False
        
        if (self.type == 'train'):
            self.target_path = 'places_train'
            self.cropped_path = 'places_train_cropped'
            self.crop_path = 'places_train_crop'
        elif (self.type == 'valid'):
            self.target_path = 'places_valid'
            self.cropped_path = 'places_valid_cropped'
            self.crop_path = 'places_valid_crop'
        elif (self.type == 'test'):
            self.target_path = 'places_test'
            self.cropped_path = 'places_test_cropped'
            self.crop_path = 'places_test_crop'
        else:
            print("error, no such data_type")
            
            
        if (self.shuffle == True):
            # ======================target images=============================
            path = os.path.join(self.folder_path, self.target_path)
            
            images = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.jpg'): images.append(os.path.join(root, file))
            print(len(images))
            #setting the shuffle:=============================================
            order = np.arange(len(images))
            order = list(order)
            if (seed == True):
                random.seed(42)
            random.shuffle(order)
            self.order = order
            #=================================================================
            
            tmp = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))
            for i in range(self.batch_num):
                for j in range(self.batch_size):
                    img = Image.open(images[self.order[i*self.batch_size + j]])
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
                    
            tmp = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))
            for i in range(self.batch_num):
                for j in range(self.batch_size):
                    img = Image.open(images[self.order[i*self.batch_size + j]])
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
            
            tmp = np.zeros((self.batch_num, self.batch_size, 28, 28, 3))
            for i in range(self.batch_num):
                for j in range(self.batch_size):
                    img = Image.open(images[self.order[i*self.batch_size + j]])
                    img_temp=np.array(img)
                    tmp[i,j]  =  img_temp
                        
            self.crop_images = tmp
            # =====================================result images===========================
            # preparing the space
            self.result_images = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))
            
           
            #===================================reading in the csv
            tmp = np.zeros(((len(images)), 8))
            csv_path = 'out_' + self.type + '.csv'
            csv_path = os.path.join(self.folder_path, csv_path)
            with open(csv_path, "r") as infile:
                r = csv.reader(infile)
                i = 0
                for i in range(len(images)):
                    row = next(r)
                    newrow = row[1:]
                    newrow = list(map(int, newrow))
                    newrow = np.array(newrow)
                    tmp[i] = newrow
                    #i += 1
            tmp = tmp.astype('int32')
            self.longcsv = tmp
            
            tmp = np.zeros((self.batch_num, self.batch_size, 8))
            for i in range(self.batch_num):
                for j in range(self.batch_size):
                    csv_temp = self.longcsv[self.order[i*self.batch_size + j]]
                    tmp[i,j]  =  csv_temp
            
            tmp = tmp.astype('int32')
            self.csv = tmp
                    
            '''
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
            '''
            
        else:
           # ======================target images=====================
            path = os.path.join(self.folder_path, self.target_path)
            
            images = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.jpg'): images.append(os.path.join(root, file))
            print(len(images))
            
            
            tmp = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))
            for i in range(self.batch_num):
                for j in range(self.batch_size):
                    img = Image.open(images[i*self.batch_size + j])
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
                    
            tmp = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))
            for i in range(self.batch_num):
                for j in range(self.batch_size):
                    img = Image.open(images[i*self.batch_size + j])
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
            
            tmp = np.zeros((self.batch_num, self.batch_size, 28, 28, 3))
            for i in range(self.batch_num):
                for j in range(self.batch_size):
                    img = Image.open(images[i*self.batch_size + j])
                    img_temp=np.array(img)
                    tmp[i,j]  =  img_temp
                        
            self.crop_images = tmp
            # =====================================result images===========================
            # preparing the space
            self.result_images = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))
            
            #===================================reading in the csv
            tmp = np.zeros(((len(images)), 8))
            csv_path = 'out_' + self.type + '.csv'
            csv_path = os.path.join(self.folder_path, csv_path)
            with open(csv_path, "r") as infile:
                r = csv.reader(infile)
                i = 0
                for i in range(len(images)):
                    row = next(r)
                    newrow = row[1:]
                    newrow = list(map(int, newrow))
                    newrow = np.array(newrow)
                    tmp[i] = newrow
                    #i += 1
            tmp = tmp.astype('int32')
            self.longcsv = tmp
            
            tmp = np.zeros((self.batch_num, self.batch_size, 8))
            for i in range(self.batch_num):
                for j in range(self.batch_size):
                    csv_temp = self.longcsv[i*self.batch_size + j]
                    tmp[i,j]  =  csv_temp
            
            tmp = tmp.astype('int32')
            self.csv = tmp
            
            '''
            csv_path = 'out_' + self.type + '.csv'
            csv_path = os.path.join(self.folder_path, csv_path)
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
            '''     
        #important parts
        # self.batch_num
        # self.batch_size
        # self.target_images
        # self.cropped_images
        # self.crop_images
        # self.result_images
        # self.csv
        
    def reset(self):
        
        path = os.path.join(self.folder_path, self.target_path)
        
        images = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.jpg'): images.append(os.path.join(root, file))
        
        
        if ((len(images)-self.end_point) > self.length ):
            if (self.shuffle == True):
                
                # ======================target images=====================
                path = os.path.join(self.folder_path, self.target_path)
            
                images = []
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.jpg'): images.append(os.path.join(root, file))
                print(len(images))
                    
                tmp = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))
                for i in range(self.batch_num):
                    for j in range(self.batch_size):
                        img = Image.open(images[self.order[i*self.batch_size + j + self.end_point]])
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
                    
                tmp = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))
                for i in range(self.batch_num):
                    for j in range(self.batch_size):
                        img = Image.open(images[self.order[i*self.batch_size + j + self.end_point]])
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
                    
                tmp = np.zeros((self.batch_num, self.batch_size, 28, 28, 3))
                for i in range(self.batch_num):
                    for j in range(self.batch_size):
                        img = Image.open(images[self.order[i*self.batch_size + j + self.end_point]])
                        img_temp=np.array(img)
                        tmp[i,j]  =  img_temp
                        
                self.crop_images = tmp
                # =====================================result images===========================
                # preparing the space
                self.result_images = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))
                #===================================reading in the csv
               
                tmp = np.zeros((self.batch_num, self.batch_size, 8))
                for i in range(self.batch_num):
                    for j in range(self.batch_size):
                        csv_temp = self.longcsv[self.order[i*self.batch_size + j + self.end_point]]
                        tmp[i,j]  =  csv_temp
            
                tmp = tmp.astype('int32')
                self.csv = tmp
               
                self.end_point = self.end_point + self.length
                
            else:
            
                # ======================target images=====================
                path = os.path.join(self.folder_path, self.target_path)
            
                images = []
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.jpg'): images.append(os.path.join(root, file))
                print(len(images))
                    
                tmp = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))
                for i in range(self.batch_num):
                    for j in range(self.batch_size):
                        img = Image.open(images[i*self.batch_size + j + self.end_point])
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
                    
                tmp = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))
                for i in range(self.batch_num):
                    for j in range(self.batch_size):
                        img = Image.open(images[i*self.batch_size + j + self.end_point])
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
                    
                tmp = np.zeros((self.batch_num, self.batch_size, 28, 28, 3))
                for i in range(self.batch_num):
                    for j in range(self.batch_size):
                        img = Image.open(images[i*self.batch_size + j + self.end_point])
                        img_temp=np.array(img)
                        tmp[i,j]  =  img_temp
                        
                self.crop_images = tmp
                # =====================================result images===========================
                # preparing the space
                self.result_images = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))
                #===================================reading in the csv
                tmp = np.zeros((self.batch_num, self.batch_size, 8))
                for i in range(self.batch_num):
                    for j in range(self.batch_size):
                        csv_temp = self.longcsv[i*self.batch_size + j + self.end_point]
                        tmp[i,j]  =  csv_temp
            
                tmp = tmp.astype('int32')
                self.csv = tmp
                
                self.end_point = self.end_point + self.length
        else:
            print("not enough pictures for a next group of batches, try changing the batch_num is it is possible")
    
    
    def change_batch_num(self,new_batch_num): # only do this before reset!
        self.batch_num = new_batch_num
        self.length = self.batch_num*self.batch_size
        
    def change_batch_size(self,new_batch_size): # only do this before reset!
        self.batch_size = new_batch_size
        self.length = self.batch_num*self.batch_size
    
    def make_masks(self):
        
        masks = np.zeros((self.batch_num, self.batch_size, 64, 64))
        for i in range(self.batch_num):
            for j in range(self.batch_size):
                x1,y1,x2,y2 = self.csv[i][j][0:4]
                maskx = x2-x1 + 1 
                masky = y2-y1 + 1
                mask = np.ones((masky, maskx))
                padr = 63 - x2
                padl = x1
                padu = 63 - y2
                padd = y1
                mask = np.pad(mask,((padu, padd),(padl, padr)))
                if (len(mask) == 64 and len(mask[0]) == 64):
                    masks[i,j] = mask
                #print(len(mask), len(mask[0]))
        if (self.made_masks == False):
            self.cropped_images = np. insert(self.cropped_images, 3, masks, axis = 4)
            self.made_masks = True
                    
                
    
    
    
        