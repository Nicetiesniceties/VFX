#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os
import random as rd
import matplotlib.pyplot as plt


# In[16]:


def readImages(directory):
    images_name = []
    images = []
    images_r, images_g, images_b = [], [], []
    images_rgb = [images_r, images_g, images_b]
    for filename in os.listdir(directory):
        images_name.append( directory + '/' + filename)
    images_name = sorted(images_name)
    for i in images_name:
        images.append(cv2.imread(i))
        b, g, r = cv2.split(images[-1])
        images_r.append(np.array(r))
        images_g.append(np.array(b))
        images_b.append(np.array(g))
        # cv2.imwrite("img.png", cv2.merge((b,g,r)))
        # images_rgb.append(cv2.cvtColor(images[-1], cv2.COLOR_BGR2RGB))
    images_r, images_g, images_b = np.array(images_r), np.array(images_g), np.array(images_b)
    return images, images_rgb
images, images_rgb = readImages("testing_images")


# In[17]:


def sampleZ(images_1c, MAX_intensity):
    images = images_1c
    intensity_range = MAX_intensity + 1
    Z = np.zeros((intensity_range, len(images)), dtype = int)
    # src_img = images[rd.randrange(len(images))]
    src_img = images[0]
    for intensity in range(intensity_range):
        rows, cols = np.where(src_img == intensity)
        if(len(rows) == 0):
            rows, cols = np.where(images[len(images) // 2 + 1] == intensity)
            if(len(rows) == 0):
                continue
        idx = rd.randrange(len(rows))
        r, c = rows[idx], cols[idx]
        for img_idx in range(len(images)):
            Z[intensity, img_idx] = images[img_idx][r][c]
    return Z
Z_r = sampleZ(images_rgb[0], 255)
Z_g = sampleZ(images_rgb[1], 255)
Z_b = sampleZ(images_rgb[2], 255)
print(Z_r)


# In[4]:


def intensity_weighting(pix_value, MAX_intensity):
    return(min(pix_value, MAX_intensity - pix_value))


# In[5]:


def intensityAdjustment(image, template):
    g, b, r = cv2.split(image)
    tb, tg, tr = cv2.split(template)
    b *= np.average(tb) / np.nanmean(b)
    g *= np.average(tg) / np.nanmean(g)
    r *= np.average(tr) / np.nanmean(r)
    # image = np.average(template) / np.nanmean(image) * image
    image = cv2.merge((g,b,r))
    return image
# intensityAdjustment(output, images[len(images) // 2])


# In[15]:


from tqdm import tqdm_notebook as tqdm
def radiance_map_1c(images_1c, shutter_times, response_curve, weighting_function):
    images = images_1c;
    rad_map = np.zeros(images[0].shape);
    
    for i in tqdm(range(len(images[0]))):
        for j in range(len(images[0][0])):
            g = np.array( [response_curve[images[n][i][j]] for n in range(len(images))] )
            w = np.array( [weighting_function(images[n][i][j], 255) for n in range(len(images))] )
            SumW = np.sum(w)
            if SumW > 0:
                rad_map[i][j] = np.sum(w * (g - shutter_times) / SumW)
            else:
                idx = rd.randrange(len(images))
                rad_map[i][j] = g[idx] - shutter_times[idx]
    return rad_map


# In[7]:


def show_images(images):
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    for p in range(len(images)):
        r = int(p / 4)
        c = int(p % 4)
        axes[r][c].imshow(images[p])
    plt.show()
show_images(images)


# In[21]:


shutter_times = np.array([1/160, 1/125, 1/80, 1/60, 1/40, 1/15], dtype = float)
# shutter_list = [ 2 ** 10, 2 ** 9, 2 ** 8, 2 ** 7, 
#                  2 ** 6, 2 ** 5, 2 ** 4, 2 ** 3, 
#                  2 ** 2, 2 ** 1, 2 ** (-1), 2 ** (-2), 
#                  2 ** (-3), 2 ** (-4), 2 ** (-5), 2 ** (-6)]
# shutter_times = np.array(shutter_list, dtype=np.float32)
def computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, weighting_function):
    z_min, z_max = 0, 255
    intensity_range = 255  # difference between min and max possible pixel value for uint8
    num_samples = intensity_samples.shape[0]
    num_images = len(log_exposures)

    # NxP + [(Zmax-1) - (Zmin + 1)] + 1 constraints; N + 256 columns
    mat_A = np.zeros((num_images * num_samples + intensity_range, num_samples + intensity_range + 1), dtype=np.float64)
    mat_b = np.zeros((mat_A.shape[0], 1), dtype=np.float64)

    # 1. Add data-fitting constraints:
    k = 0
    for i in range(num_samples):
        for j in range(num_images):
            z_ij = intensity_samples[i, j]
            w_ij = weighting_function(z_ij, 255)
            mat_A[k, z_ij] = w_ij
            mat_A[k, (intensity_range + 1) + i] = -w_ij
            mat_b[k, 0] = w_ij * log_exposures[j]
            k += 1
    
    # 2. Add smoothing constraints:
    for z_k in range(z_min + 1, z_max):
        w_k = weighting_function(z_k, 255)
        mat_A[k, z_k - 1] = w_k * smoothing_lambda
        mat_A[k, z_k    ] = -2 * w_k * smoothing_lambda
        mat_A[k, z_k + 1] = w_k * smoothing_lambda
        k += 1

    # 3. Add color curve centering constraint:
    mat_A[k, (z_max - z_min) // 2] = 1
    
    inv_A = np.linalg.pinv(mat_A)
    x = np.dot(inv_A, mat_b)
   
    g = x[0: intensity_range + 1]
    return g[:, 0]
R_r = computeResponseCurve(Z_r, shutter_times, 2, intensity_weighting)
R_g = computeResponseCurve(Z_g, shutter_times, 2, intensity_weighting)
R_b = computeResponseCurve(Z_b, shutter_times, 2, intensity_weighting)


# In[ ]:


RR = radiance_map_1c(images_rgb[0], shutter_times, R_r, intensity_weighting)
GG = radiance_map_1c(images_rgb[1], shutter_times, R_g, intensity_weighting)
BB = radiance_map_1c(images_rgb[2], shutter_times, R_b, intensity_weighting)


# In[ ]:


gamma = 1
R = cv2.normalize(RR, np.array([]), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
G = cv2.normalize(GG, np.array([]), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
B = cv2.normalize(BB, np.array([]), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
def globalToneMapping(image, gamma):
    image_corrected = cv2.pow(image/255., 1.0/gamma)
    return image_corrected
# R = globalToneMapping(R, gamma)
# G = globalToneMapping(G, gamma)
# B = globalToneMapping(B, gamma)
R = cv2.normalize(R, np.array([]), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
G = cv2.normalize(G, np.array([]), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
B = cv2.normalize(B, np.array([]), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
output = cv2.merge((G,B,R))
output = intensityAdjustment(output, images[5])
cv2.imwrite("img1.png", output)


# In[ ]:




