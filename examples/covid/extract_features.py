# =============================================================================
# feature extraction to create feature pool
# =============================================================================
import skimage.io as io
from skimage.transform import  rescale,resize
from skimage.util import img_as_uint,img_as_ubyte
from skimage.color import rgb2gray
from skimage import exposure
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
from utils import*
# =============================================================================
# source dir and output file name
# =============================================================================
class_name='pneumonia'#'covid' or 'normal' or 'pneumonia'
source_dir='./covid_big_dset2/'+class_name
output_file_name="./covid_big_dset2/"+class_name
# =============================================================================
# set labels
# =============================================================================
if class_name=='normal':
    label=0
elif class_name=='covid':
    label=1
else:
    label=2
# =============================================================================
# start
# =============================================================================
image_list=os.listdir(source_dir)#list of images
print("Num of images: ", len(image_list))
feature_pool=np.empty([1,252])#feature pool
for idx,img_name in enumerate(image_list):
    print(idx)
    img=io.imread(os.path.join(source_dir,img_name))
    img_rescaled=(img-np.min(img))/(np.max(img)-np.min(img)) 
    
    texture_features=compute_14_features(img_rescaled)#texture features
    
    fft_map=np.fft.fft2(img_rescaled)
    fft_map = np.fft.fftshift(fft_map)
    fft_map = np.abs(fft_map)
    YC=int(np.floor(fft_map.shape[1]/2)+1)
    fft_map=fft_map[:,YC:int(np.floor(3*YC/2))]
    fft_features=compute_14_features(fft_map)#FFT features
    
    wavelet_coeffs = pywt.dwt2(img_rescaled,'sym4')
    cA1, (cH1, cV1, cD1) = wavelet_coeffs
    wavelet_coeffs = pywt.dwt2(cA1,'sym4')
    cA2, (cH2, cV2, cD2) = wavelet_coeffs#wavelet features
    wavelet_features=np.concatenate((compute_14_features(cA1), compute_14_features(cH1),compute_14_features(cV1),compute_14_features(cD1)
    ,compute_14_features(cA2), compute_14_features(cH2),compute_14_features(cV2),compute_14_features(cD2)), axis=0)
    
    
    gLDM1,gLDM2,gLDM3,gLDM4=GLDM(img_rescaled,10)#GLDM in four directions
    gldm_features=np.concatenate((compute_14_features(gLDM1), compute_14_features(gLDM2),
                                  compute_14_features(gLDM3),compute_14_features(gLDM4)), axis=0)
    
    
    glcms =greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])#GLCM in four directions
    glcm_features=np.concatenate((compute_14_features(im2double(glcms[:, :, 0, 0])), 
                                  compute_14_features(im2double(glcms[:, :, 0, 1])),
                                  compute_14_features(im2double(im2double(glcms[:, :, 0, 2]))),
                                  compute_14_features(glcms[:, :, 0, 3])), axis=0)
    
    feature_vector=np.concatenate((texture_features,fft_features,wavelet_features,gldm_features,glcm_features), axis=0).reshape(1,252)#merge to create a feature vector of 252
    feature_pool=np.concatenate((feature_pool,feature_vector), axis=0)


feature_pool=np.delete(feature_pool, 0, 0)
feature_pool=np.concatenate((feature_pool,label*np.ones(len(feature_pool)).reshape(len(feature_pool),1)), axis=1)#add label to the last column   
sio.savemat(output_file_name+'.mat', {class_name: feature_pool})#save the created feature pool as a mat file  
