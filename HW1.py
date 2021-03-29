#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os


# In[2]:


get_ipython().system('pip install pydicom')
import pydicom as pyd
get_ipython().system('pip install opencv-contrib-python')
import cv2


# In[86]:


get_ipython().run_cell_magic('cmd', '', 'tar xvf CT_chest_scans.zip')


# In[3]:


def printDicomInfo(filepath):
    ds = pyd.dcmread(filepath)
    print(ds)


# In[4]:


# sort: use InstanceNumber
def sort(slices):
    sorted_slices = (sorted(slices, key = lambda s: s.InstanceNumber))

    return sorted_slices


# In[5]:


def dicom2HU(file):
    image = file.pixel_array
    image_HU = np.copy(image)

    #deal with padding value
    image_HU[image_HU == -2000] = 0
    #transport to HU
    intercept = file.RescaleIntercept
    slope = file.RescaleSlope

    image_HU = slope * image_HU + intercept

    #compute mean, var, max, min
    print("mean: HU ", np.mean(image_HU), "origin ", np.mean(image))
    print("standard deviation: HU ", np.sqrt(np.var(image_HU)), "origin ",np.sqrt(np.var(image)))
    print("max: HU ", np.max(image_HU), "origin ",np.max(image))
    print("min: HU ", np.min(image_HU), "origin ",np.min(image))
    return image_HU


# In[6]:


def normalize(imgs):
    img_norm = np.copy(imgs)
    for i in range(len(imgs)):
        img = imgs[i]
        MIN_BOUND = np.min(img)
        MAX_BOUND = np.max(img)
        img_norm[i] = (img - MIN_BOUND) / float(MAX_BOUND - MIN_BOUND)
        
        
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(imgs[i], cmap=plt.cm.gray)
    plt.show()
    return img_norm


# In[7]:


def localMedianThreshold(img):
    threshold = np.median(img)
    image = img.copy()

    plt.hist(image.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.axvline(threshold , color='k', linewidth=1)
    plt.show()

    image[image > threshold]= 2000
    image[image < threshold]= -2000
    plt.imshow(image, cmap=plt.cm.gray, vmin = -2000, vmax= 2000)
    plt.show()


# In[8]:


def localMeanThreshold(img):
    threshold = np.mean(img)
    image = img.copy()

    plt.hist(image.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.axvline(threshold , color='k', linewidth=1)
    plt.show()

    image[image > threshold]= 2000
    image[image < threshold]= -2000
    plt.imshow(image, cmap=plt.cm.gray, vmin = -2000, vmax= 2000)
    plt.show()
    return image


# In[9]:


# Otsu's thresholding
def OstuThreshold(img):
    
    image = img.copy()
    MIN_BOUND = np.min(image)
    MAX_BOUND = np.max(image)
    image = (image - MIN_BOUND) / float(MAX_BOUND - MIN_BOUND)
    image*=255
    cv2.imwrite('examples.png', image)
    image = cv2.imread('examples.png', 0)

    threshold,image1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    plt.hist(image.flatten(), bins=80, color='c')
    plt.xlabel("gray scale value")
    plt.ylabel("Frequency")
    plt.axvline(threshold , color='k', linewidth=1)
    plt.show()

    plt.imshow(image1, cmap=plt.cm.gray)
    plt.show()


# In[14]:


def saveSegmented(slices):
    if not os.path.exists("CT_chest_scans_segmented"):
        os.makedirs("CT_chest_scans_segmented")
    for i in range(len(slices)):
        file = os.listdir(path+'/'+dirList[0])[i]
        ds = pyd.read_file(path+'/'+dirList[0]+'/'+file)
        img = dicom2HU(ds)
        threshold = np.mean(img)
        img = np.where(img < threshold, 2000, -2000)
        img = img.astype(np.uint16)
        ds.PixelData = img.tobytes()
        ds.save_as("CT_chest_scans_segmented/"+file)
        


# In[13]:


if __name__ == "__main__":
    # Part1: read Dicom Information
    path = "CT_chest_scans"
    dirList = os.listdir(path)
    file = os.listdir(path+'/'+dirList[0])[0]
    filepath = path+'/'+dirList[0]+'/'+file
    printDicomInfo(filepath)
    
    #Part2: read multi Dicom files, convert to HU, and normalize
    path = "CT_chest_scans"
    dirList = os.listdir(path)
    slices = [pyd.read_file(path+'/'+dirList[0]+'/'+s) for s in os.listdir(path+'/'+dirList[0])]
    sorted_slices = sort(slices)
    
    images_HU = np.stack([s.pixel_array for s in sorted_slices])
    for i in range(len(sorted_slices)):
        images_HU[i] = dicom2HU(sorted_slices[i])
    
    images_norm = normalize(images_HU)
    
    #Part3: segmented
    plt.imshow(images_HU[20], cmap=plt.cm.gray)
    plt.show()
    localMedianThreshold(images_HU[20])
    localMeanThreshold(images_HU[20])
    OstuThreshold(images_HU[20])
    
    #Bouns: Construct 3D model
    saveSegmented(sorted_slices)

