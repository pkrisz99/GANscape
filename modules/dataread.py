# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:21:10 2020

@author: KR
"""
#imports
import os
from PIL import Image
import numpy as np
import random
import csv


class DataRead:
    def __init__(self, folder_path, data_type, batch_size, batch_num, shuffle = False, seed = False):
        self.folder_path = folder_path #the places where we have all our data
        self.type = data_type # what type of data wa have
        self.shuffle = shuffle # whether we want to shuffle
        self.seed = seed # whether we want to use a random seed
       
        self.length = batch_num*batch_size # the number of pictures
            
        self.batch_size = batch_size # batch size
        self.batch_num = batch_num # number of batches
        
        self.end_point = self.length #endpoint, how many picture did we use all together
        #self.made_masks = False
        
        # making the needed paths according to the data type
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
            
            
        if (self.shuffle == True): # if we want to shuffle
            # ======================target images=============================
            path = os.path.join(self.folder_path, self.target_path)# making the path
            
            images = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.jpg'): images.append(os.path.join(root, file))#getting the pathss to the pictures
            print(len(images)) # checking the number of images
            images.sort()
            #setting the shuffle:=============================================
            order = np.arange(len(images)) #the number of all the images
            order = list(order)
            if (seed == True): #setting a seed if it is needed
                random.seed(42)
            random.shuffle(order) # random order
            self.order = order # saving the order of the pics
            #=================================================================
            
            tmp = np.zeros((self.batch_num, self.batch_size, 64, 64, 3)) # making the space for the pictures
            for i in range(self.batch_num):
                for j in range(self.batch_size):
                    img = Image.open(images[self.order[i*self.batch_size + j]])
                    img_temp=np.array(img)
                    tmp[i,j]  =  img_temp #filling the array with the pictures
                        
            self.target_images = tmp # saving the pictures 
            # ============================train images-cropped=============================
            path = os.path.join(self.folder_path, self.cropped_path)# setting a new path
            
            images = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.jpg'): images.append(os.path.join(root, file))#getting the paths of the pictures
            print(len(images))#checking the number of images
            images.sort()
            
            tmp = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))# making space for the pictures
            for i in range(self.batch_num):
                for j in range(self.batch_size):
                    img = Image.open(images[self.order[i*self.batch_size + j]])
                    img_temp=np.array(img)
                    tmp[i,j]  =  img_temp # filling the array with the pictures
                        
            self.cropped_images = tmp # saving the pictures
            # =============================train images-crop===========================
            path = os.path.join(self.folder_path, self.crop_path) # new path
            
            images = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.jpg'): images.append(os.path.join(root, file)) # paths of the pics
            print(len(images))    # checking the number of pics
            images.sort()
            
            tmp = np.zeros((self.batch_num, self.batch_size, 28, 28, 3)) # makking space for the pics
            for i in range(self.batch_num):
                for j in range(self.batch_size):
                    img = Image.open(images[self.order[i*self.batch_size + j]])
                    img_temp=np.array(img)
                    tmp[i,j]  =  img_temp #filling the array with the pics
                        
            self.crop_images = tmp # saving the pics
            # =====================================result images===========================
            # preparing the space
            #self.result_images = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))
            
           
            #===================================reading in the csv=======================
            tmp = np.zeros(((len(images)), 8)) # making space for the csv data
            csv_path = 'out_' + self.type + '.csv'
            csv_path = os.path.join(self.folder_path, csv_path) # setting the path
            with open(csv_path, "r") as infile:
                r = csv.reader(infile)
                i = 0
                for i in range(len(images)):
                    row = next(r) # reading in the lines
                    newrow = row[1:]
                    newrow = list(map(int, newrow))
                    newrow = np.array(newrow)
                    tmp[i] = newrow #saving the rows
                    #i += 1
            tmp = tmp.astype('int32')
            self.longcsv = tmp # saving al the csv data
            
            tmp = np.zeros((self.batch_num, self.batch_size, 8)) # making space for the csv data that we need 
            for i in range(self.batch_num):
                for j in range(self.batch_size):
                    csv_temp = self.longcsv[self.order[i*self.batch_size + j]]
                    tmp[i,j]  =  csv_temp# filling the array
            
            tmp = tmp.astype('int32')
            self.csv = tmp #the part of the csv data that we actually need
                    
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
        # it is the same as before without the random order
           # ======================target images=====================
            path = os.path.join(self.folder_path, self.target_path)# set path
            
            images = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.jpg'): images.append(os.path.join(root, file))# pic paths
            print(len(images))
            images.sort()
            
            for i in range(100):
                print(images[i])
            
            
            tmp = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))# making space
            for i in range(self.batch_num):
                for j in range(self.batch_size):
                    img = Image.open(images[i*self.batch_size + j])
                    img_temp=np.array(img)
                    tmp[i,j]  =  img_temp#filling the array
                        
            self.target_images = tmp# saving the pics
            # ============================train images-cropped=============================
            path = os.path.join(self.folder_path, self.cropped_path)# new path 
            
            images = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.jpg'): images.append(os.path.join(root, file))# pic paths
            print(len(images))
            images.sort()
                    
            tmp = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))#making space
            for i in range(self.batch_num):
                for j in range(self.batch_size):
                    img = Image.open(images[i*self.batch_size + j])
                    img_temp=np.array(img)
                    tmp[i,j]  =  img_temp#fillig the array
                        
            self.cropped_images = tmp#saving the pics
            # =============================train images-crop===========================
            path = os.path.join(self.folder_path, self.crop_path)#new path
            
            images = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.jpg'): images.append(os.path.join(root, file))#ppic paths
            print(len(images)) 
            images.sort()
            
            tmp = np.zeros((self.batch_num, self.batch_size, 28, 28, 3))#making space
            for i in range(self.batch_num):
                for j in range(self.batch_size):
                    img = Image.open(images[i*self.batch_size + j])
                    img_temp=np.array(img)
                    tmp[i,j]  =  img_temp#filling the array
                        
            self.crop_images = tmp#saving the pics
            # =====================================result images===========================
            # preparing the space
            #self.result_images = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))
            
            #===================================reading in the csv======================
            tmp = np.zeros(((len(images)), 8))#making space 
            csv_path = 'out_' + self.type + '.csv'
            csv_path = os.path.join(self.folder_path, csv_path)#path for the csv
            with open(csv_path, "r") as infile:
                r = csv.reader(infile)
                i = 0
                for i in range(len(images)):
                    row = next(r)
                    newrow = row[1:]
                    newrow = list(map(int, newrow))
                    newrow = np.array(newrow)
                    tmp[i] = newrow #reading in all the lines
                    #i += 1
            tmp = tmp.astype('int32')
            self.longcsv = tmp # saving the array that contains all the csv information
            
            tmp = np.zeros((self.batch_num, self.batch_size, 8)) # making space
            for i in range(self.batch_num):
                for j in range(self.batch_size):
                    csv_temp = self.longcsv[i*self.batch_size + j]
                    tmp[i,j]  =  csv_temp #saving those lines that we need
            
            tmp = tmp.astype('int32')
            self.csv = tmp# saving the scv lines that we actually need
            
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
        
    def reset(self): # resetting changing the pictures for new ones
        
        path = os.path.join(self.folder_path, self.target_path)#setting path
        
        images = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.jpg'): images.append(os.path.join(root, file))#collecting the paths to the images
        #images.sort()
        
        if ((len(images)-self.end_point) > self.length ): # if we have enough pictures left
            if (self.shuffle == True): # cheking the shuffle
                
                # ======================target images=====================
                path = os.path.join(self.folder_path, self.target_path)# new path
            
                images = []
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.jpg'): images.append(os.path.join(root, file))#collecting the paths to the images
                print(len(images))
                images.sort()
                    
                tmp = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))# place for the new pics
                for i in range(self.batch_num):
                    for j in range(self.batch_size):
                        img = Image.open(images[self.order[i*self.batch_size + j + self.end_point]])
                        img_temp=np.array(img)
                        tmp[i,j]  =  img_temp# filling the array
                        
                self.target_images = tmp#saving the new images
                # ============================train images-cropped=============================
                path = os.path.join(self.folder_path, self.cropped_path)#new path
            
                images = []
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.jpg'): images.append(os.path.join(root, file))#pic paths
                print(len(images))
                images.sort()
                    
                tmp = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))#making space
                for i in range(self.batch_num):
                    for j in range(self.batch_size):
                        img = Image.open(images[self.order[i*self.batch_size + j + self.end_point]])
                        img_temp=np.array(img)
                        tmp[i,j]  =  img_temp# filling the array
                        
                self.cropped_images = tmp# saving the pics
                # =============================train images-crop===========================
                path = os.path.join(self.folder_path, self.crop_path)# new path
            
                images = []
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.jpg'): images.append(os.path.join(root, file))#pic paths
                print(len(images)) 
                images.sort()
                    
                tmp = np.zeros((self.batch_num, self.batch_size, 28, 28, 3))#making space
                for i in range(self.batch_num):
                    for j in range(self.batch_size):
                        img = Image.open(images[self.order[i*self.batch_size + j + self.end_point]])
                        img_temp=np.array(img)
                        tmp[i,j]  =  img_temp#filling the array
                        
                self.crop_images = tmp#saving the pics
                # =====================================result images===========================
                # preparing the space
                #self.result_images = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))
                #===================================reading in the csv
               
                tmp = np.zeros((self.batch_num, self.batch_size, 8))
                for i in range(self.batch_num):
                    for j in range(self.batch_size):
                        csv_temp = self.longcsv[self.order[i*self.batch_size + j + self.end_point]]
                        tmp[i,j]  =  csv_temp #loading in new .csv lines
            
                tmp = tmp.astype('int32')
                self.csv = tmp # saving hte csv lines
               
                self.end_point = self.end_point + self.length# setting the new endpoint
                
            else:
                #basically the same without the shuffle
                # ======================target images=====================
                path = os.path.join(self.folder_path, self.target_path)# new path
            
                images = []
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.jpg'): images.append(os.path.join(root, file))#pic paths
                print(len(images))
                images.sort()
                    
                tmp = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))#making space
                for i in range(self.batch_num):
                    for j in range(self.batch_size):
                        img = Image.open(images[i*self.batch_size + j + self.end_point])
                        img_temp=np.array(img)
                        tmp[i,j]  =  img_temp#filling array
                        
                self.target_images = tmp#saving pics
                # ============================train images-cropped=============================
                path = os.path.join(self.folder_path, self.cropped_path)#new path
            
                images = []
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.jpg'): images.append(os.path.join(root, file))#pic paths
                print(len(images))
                images.sort()
                    
                tmp = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))#making space
                for i in range(self.batch_num):
                    for j in range(self.batch_size):
                        img = Image.open(images[i*self.batch_size + j + self.end_point])
                        img_temp=np.array(img)
                        tmp[i,j]  =  img_temp#filling array
                        
                self.cropped_images = tmp#saving pics
                # =============================train images-crop===========================
                path = os.path.join(self.folder_path, self.crop_path)# new path
            
                images = []
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.jpg'): images.append(os.path.join(root, file))#pic paths
                print(len(images))    
                images.sort()
                    
                tmp = np.zeros((self.batch_num, self.batch_size, 28, 28, 3))#making space
                for i in range(self.batch_num):
                    for j in range(self.batch_size):
                        img = Image.open(images[i*self.batch_size + j + self.end_point])
                        img_temp=np.array(img)
                        tmp[i,j]  =  img_temp#filling array
                        
                self.crop_images = tmp#saving pics
                # =====================================result images===========================
                # preparing the space
                #self.result_images = np.zeros((self.batch_num, self.batch_size, 64, 64, 3))
                #===================================reading in the csv
                tmp = np.zeros((self.batch_num, self.batch_size, 8))
                for i in range(self.batch_num):
                    for j in range(self.batch_size):
                        csv_temp = self.longcsv[i*self.batch_size + j + self.end_point]
                        tmp[i,j]  =  csv_temp
            
                tmp = tmp.astype('int32')
                self.csv = tmp#loading the new csv lines
                
                self.end_point = self.end_point + self.length#setting the new endpoint
        else:
            print("not enough pictures for a next group of batches, try changing the batch_num is it is possible")
    
    
    def change_batch_num(self,new_batch_num): # only do this before reset!
        self.batch_num = new_batch_num # changing the number of batches
        self.length = self.batch_num*self.batch_size # we have to changes the length as well
        
    def change_batch_size(self,new_batch_size): # only do this before reset!
        self.batch_size = new_batch_size # changing the batch size
        self.length = self.batch_num*self.batch_size # we have to changes the length as well
    
    '''
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
                    
     '''           
    
    
    
        