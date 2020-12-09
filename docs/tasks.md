# 1. CIFAR10 dataset
## 1.1. Information and installation
### 1.1.1. Information about the dataset
* The CIFAR-10 dataset consists of 60000 32x32x3 colour images in 10 classes, with 6000 images per class. 
* The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks
* Input for NN are raw 32x32 3 channels GRB images
* NN output is distribution of probabilities for each class i.e. 10 values that sums up to 1


* Code folder: [here](colearn_examples/cifar10)
* Invoke parameter: -t CIFAR10
###1.1.2. Requirements
* Cifar dataset is loaded from tensorflow.keras.datasets.cifar10 and no stored data are required

## 1.2. Models
### 1.2.1. CIFAR10Conv Keras model
```
_________________________________________________________________
Layer (type)                    Output Shape        Param #   
=================================================================
Input (InputLayer)              (32, 32, 3)         0             
_________________________________________________________________
Conv1_1 (Conv2D)                (32, 32, 64)        1792          
bn1_1 (BatchNormalization)      (32, 32, 64)        256           
Conv1_2 (Conv2D)                (32, 32, 64)        36928         
bn1_2 (BatchNormalization)      (32, 32, 64)        256           
pool1 (MaxPooling2D)            (16, 16, 64)        0             
_________________________________________________________________
Conv2_1 (Conv2D)                (16, 16, 128        73856         
bn2_1 (BatchNormalization)      (16, 16, 128        512           
Conv2_2 (Conv2D)                (16, 16, 128        147584    
bn2_2 (BatchNormalization)      (16, 16, 128        512           
pool2 (MaxPooling2D)            (8, 8, 128)         0             
_________________________________________________________________
Conv3_1 (Conv2D)                (8, 8, 256)         295168    
bn3_1 (BatchNormalization)      (8, 8, 256)         1024          
Conv3_2 (Conv2D)                (8, 8, 256)         590080    
bn3_2 (BatchNormalization)      (8, 8, 256)         1024          
Conv3_3 (Conv2D)                (8, 8, 256)         590080    
bn3_3 (BatchNormalization)      (8, 8, 256)         1024          
_________________________________________________________________
flatten (Flatten)               (16384)             0             
fc1 (Dense)                     (100)               1638500   
fc2 (Dense)                     (10)                1010          
=================================================================
Total params: 3,379,606
Trainable params: 3,377,302
Non-trainable params: 2,304
_________________________________________________________________
```

###1.2.2. CIFAR10Conv2 Keras model
```
_________________________________________________________
Layer (type)                Output Shape        Param #   
=========================================================
Input (InputLayer)          (32, 32, 3)         0             
_________________________________________________________
Conv1_1 (Conv2D)            (32, 32, 32)        896           
Conv1_2 (Conv2D)            (32, 32, 32)        9248          
pool1 (MaxPooling2D)        (16, 16, 32)        0             
_________________________________________________________
Conv2_1 (Conv2D)            (16, 16, 64)        18496         
Conv2_2 (Conv2D)            (16, 16, 64)        36928         
pool2 (MaxPooling2D)        (8, 8, 64)          0             
_________________________________________________________
Conv3_1 (Conv2D)            (8, 8, 128)         73856         
Conv3_2 (Conv2D)            (8, 8, 128)         147584    
pool3 (MaxPooling2D)        (4, 4, 128)         0             
_________________________________________________________
flatten (Flatten)           (2048)              0             
fc1 (Dense)                 (128)               262272    
fc2 (Dense)                 (10)                1290          
=========================================================
Total params: 550,570
Trainable params: 550,570
Non-trainable params: 0
_________________________________________________________
```
###1.2.3. CIFAR10Resnet50 Keras model
```
________________________________________________________
Layer (type)                 Output Shape     Param #   
========================================================
Input (InputLayer)           (32, 32, 3)]     0             
________________________________________________________
resnet50 (Model)             (1, 1, 2048)     23587712  
________________________________________________________
Global_average_pooling2d     (2048)           0             
flatten (Flatten)            (2048)           0             
fc1 (Dense)                  (10)             20490         
========================================================
Total params: 23,608,202
Trainable params: 23,555,082
Non-trainable params: 53,120
________________________________________________________
```
#2. Covid X-RAY dataset
##2.1. Information and installation
###2.1.1. Information about the dataset
* The Covid X-Ray dataset consists of grayscale images, there are 478 covid images and 203 normal images. 
* To increase the number of images normal/pneumonia dataset is added
* Final dataset, which is a combination of two previously mentioned datasets, contains 1434 images, 478 images for each class.
* Images are cropped and resized to 512x512 pixel and spatial domain (Texture, GLDM, GLCM) and frequency domain (FFT and Wavelet) features are used to create 256 dimensional vector representation of each image. PCA is applied after to reduce dimensionality to 64 values which represents the first 64 highest eigenvalues of the covariance matrix. 
* Input for NN are 64 values for each image
* NN output is distribution of probabilities for each class i.e. 3 values
* Code folder: [here](colearn_examples/covid_xray)
* Invoke parameter: -t COVID
###2.1.2 Requirements
* Download Covid dataset: [here](https://github.com/ieee8023/COVID-chestxray-dataset)
* Download pneumonia dataset: [here](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

##2.2. Models
###2.2.1. Covid XRAY Keras model
```
_________________________________________________________
Layer (type)              Output Shape        Param #   
=========================================================
input_1 (InputLayer)      (64)                0             
_________________________________________________________
dense (Dense)             (128)               8320          
dropout (Dropout)         (128)               0             
_________________________________________________________
dense_1 (Dense)           (16)                2064          
dropout_1 (Dropout)       (16)                0             
_________________________________________________________
dense_2 (Dense)           (3)                 51            
=========================================================
Total params: 10,435
Trainable params: 10,435
Non-trainable params: 0
_________________________________________________________
```
#3. FRAUD dataset
##3.1. Information and installation
###3.1.1. Information about the dataset
* EEE-CIS Fraud Detection, contains multiple files with credit card transactions
* Raw dataset files are automatically merged and pre-processed and input files for neural network are created
   -  X.csv with data - has 431 values for each transaction
   -  Y.csv with labels - v has 1 value for each transaction
       + 0 = not a fraud
       + 1 = fraud 

* Code folder: [here](colearn_examples/fraud)
* Invoke parameter: -t FRAUD
###3.1.2 Requirements
* Download dataset: [here](https://www.kaggle.com/c/ieee-fraud-detection)
##3.2. Models
###3.2.1. FraudDense1 Keras model
```
_________________________________________________________
Layer (type)             Output Shape          Param #   
=========================================================
Input (InputLayer)       (431)                 0             
_________________________________________________________
dense (Dense)            (512)                 221184    
Batch_normalization      (512)                 2048          
_________________________________________________________
dense_1 (Dense)          (512)                 262656    
Batch_normalization_1    (512)                 2048          
_________________________________________________________
dense_2 (Dense)          (512)                 262656    
Batch_normalization_2    (512)                 2048          
_________________________________________________________
fc1 (Dense)              (1)                   513           
=========================================================
Total params: 753,153
Trainable params: 750,081
Non-trainable params: 3,072
_________________________________________________________
```
###3.2.2. FraudSVM Scikit-learn model


* Model is defined as SGDClassifier(max_iter=1, verbose=0, loss="modified_huber")
  - Which is support vector machine linear classifier
#4. MNIST
##4.1. Information and installation
###4.1.1. Information about the dataset
* This is a dataset of 70,000 28x28x1 grayscale images of the 10 digits
* Input for NN are raw 28x28 1 channel images
* NN output is distribution of probabilities for each class i.e. 10 values that sums up to 1


* Code folder: [here](colearn_examples/mnist)
* Invoke parameter: -t MNIST
###4.1.2 Requirements
* MNIST dataset is loaded from tensorflow.keras.datasets.cifar10 and no stored data are required


##4.2. Models
###4.2.1. MNISTConv Keras model
```
_________________________________________________________
Layer (type)                   Output Shape       Param #   
=========================================================
Input (InputLayer)             (28, 28, 1)        0             
_________________________________________________________
Conv1_1 (Conv2D)               (28, 28, 64)       640           
bn1 (BatchNormalization)       (28, 28, 64)       256           
pool1 (MaxPooling2D)           (14, 14, 64)       0             
_________________________________________________________
Conv2_1 (Conv2D)               (14, 14, 128)      73856         
bn4 (BatchNormalization)       (14, 14, 128)      512           
pool2 (MaxPooling2D)           (7, 7, 128)        0             
_________________________________________________________
flatten (Flatten)              (6272)             0             
fc1 (Dense)                    (10)               62730         
=========================================================
Total params: 137,994
Trainable params: 137,610
Non-trainable params: 384
_________________________________________________________
```
###4.2.2. MNIST Pytorch model
```
---------------------------------------------------------
Layer (type)           Output Shape             Param #
=========================================================
Input                  [28,28,1]                0
Conv2d-1               [20, 24, 24]             520
Conv2d-2               [50, 8, 8]               25,050
Linear-3               [500]                    400,500
Linear-4               [10]                     5,010
=========================================================
Total params: 431,080
Trainable params: 431,080
Non-trainable params: 0
---------------------------------------------------------
```
###4.2.3. MNISTSupermini Keras model
```
________________________________________________________________________________________
Layer (type)                Output Shape    Param #      Connected to                         
========================================================================================
input_1 (InputLayer)        (28, 28, 1)     0                                                
________________________________________________________________________________________
conv2d (Conv2D)             (26, 26, 8)     80           input_1[0][0]                        
Batch_normalization         (26, 26, 8)     32           conv2d[0][0]                         
Max_pooling2d               (13, 13, 8)     0            batch_normalization[0][0]            
dropout (Dropout)           (13, 13, 8)     0            max_pooling2d[0][0]                  
________________________________________________________________________________________
Separable_conv2d            (11, 11, 26)    306          dropout[0][0]                        
batch_normalization_1       (11, 11, 26)    104          separable_conv2d[0][0]               
dropout_1 (Dropout)         (11, 11, 26)    0            batch_normalization_1[0][0]          
________________________________________________________________________________________
Separable_conv2d_1          (11, 11, 26)    936          dropout_1[0][0]                      
                                                         dropout_2[0][0]                      
                                                         dropout_3[0][0]                      
________________________________________________________________________________________
Batch_normalization_2        (11, 11, 26)   104          separable_conv2d_1[0][0]             
dropout_2 (Dropout)          (11, 11, 26)   0            batch_normalization_2[0][0]          
________________________________________________________________________________________
Batch_normalization_3        (11, 11, 26)   104          separable_conv2d_1[1][0]             
dropout_3 (Dropout)          (11, 11, 26)   0            batch_normalization_3[0][0]          
________________________________________________________________________________________
Batch_normalization_4        (11, 11, 26)   104          separable_conv2d_1[2][0]             
dropout_4 (Dropout)          (11, 11, 26)   0            batch_normalization_4[0][0]          
________________________________________________________________________________________
Global_average_pooling2d     (26)           0            dropout_4[0][0]                      
dense (Dense)                (16)           432          global_average_pooling2d[0][0]   
Batch_normalization_5        (16)           64           dense[0][0]                          
dropout_5 (Dropout)          (16)           0            batch_normalization_5[0][0]          
dense_1 (Dense)              (10)           170          dropout_5[0][0]                      
========================================================================================
Total params: 2,436
Trainable params: 2,180
Non-trainable params: 256
________________________________________________________________________________________
```

#5. Pneumonia XRAY
##5.1. Information and installation
###5.1.1. Information about the dataset
   * The Chest X-Ray Images (Pneumonia) dataset consists of 5856 grayscale images of various sizes in 2 classes (normal/pneumonia). 
   * Labels are determined by folder name - NORMAL or PNEUMONIA
   * Input for NN are raw resized 128x128 1 channel images
   * NN output is distribution of probabilities for each class i.e. 2 values


   * Code folder: [here](colearn_examples/xray)
   * Invoke parameter: -t XRAY
###5.1.2 Requirements
   * Download dataset: [here](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
##5.2. Models
###5.2.1. XraySupermini Keras model#
```
_________________________________________________________________
Layer (type)                     Output Shape            Param #   
=================================================================
Input (InputLayer)               [(128, 128, 1)]         0             
_________________________________________________________________
Conv1_1 (Conv2D)                 (128, 128, 32)          320           
_________________________________________________________________
bn1 (BatchNormalization)         (128, 128, 32)          128           
_________________________________________________________________
pool1 (MaxPooling2D)             (32, 32, 32)            0             
_________________________________________________________________
Conv2_1 (Conv2D)                 (32, 32, 64)            18496         
_________________________________________________________________
bn2 (BatchNormalization)         (32, 32, 64)            256           
_________________________________________________________________
Global_max_pooling2d             (64)                    0             
_________________________________________________________________
fc1 (Dense)                      (1)                     65            
=================================================================
Total params: 19,265
Trainable params: 19,073
Non-trainable params: 192
_________________________________________________________________
```
###5.2.2. XrayResnet50 Keras model
```
_________________________________________________________________
Layer (type)                     Output Shape            Param #   
=================================================================
Input (InputLayer)               [(128, 128, 1)]         0             
_________________________________________________________________
resnet50 (Model)                 (4, 4, 2048)            23581440  
_________________________________________________________________
global_average_pooling2d         (2048)                  0             
_________________________________________________________________
flatten (Flatten)                (2048)                  0             
_________________________________________________________________
fc1 (Dense)                      (1)                     2049          
=================================================================
Total params: 23,583,489
Trainable params: 23,530,369
Non-trainable params: 53,120
_________________________________________________________________
```
###5.2.3. XrayPretrainedResnet50 Keras model
```
_____________________________________________________________________________________
Layer (type)                Output Shape    Param #   Connected to                
=====================================================================================
Input (InputLayer)          (128, 128, 1)   0                                       
_____________________________________________________________________________________
concatenate (Concatenate)   (128, 128, 3)   0         Input[0][0]                                                                                     Input[0][0]                                                                                    Input[0][0]                 
_____________________________________________________________________________________
tf_op_layer_mul             (128, 128, 3)  0          concatenate[0][0]           
tf_op_layer_strided_slice   (128, 128, 3)  0          tf_op_layer_mul[0][0]       
tf_op_layer_BiasAdd         (128, 128, 3)  0          tf_op_layer_strided_slice[0][0]  
_____________________________________________________________________________________
resnet50 (Model)            (4, 4, 2048)   23587712   tf_op_layer_BiasAdd[0][0]            
_____________________________________________________________________________________
global_average_pooling2d    (2048)         0          resnet50[1][0]              
flatten (Flatten)           (2048)         0          global_average_pooling2d[0][0]   
_____________________________________________________________________________________
fc1 (Dense)                 (1)            2049       flatten[0][0]              
=====================================================================================
Total params: 23,589,761
Trainable params: 23,536,641
Non-trainable params: 53,120
_____________________________________________________________________________________
```
###5.2.4. XrayDropout Keras model
```
_________________________________________________________________
Layer (type)                     Output Shape            Param #   
=================================================================
Input (InputLayer)               [(128, 128, 1)]         0             
_________________________________________________________________
Conv1_1 (Conv2D)                 (128, 128, 128)         1280          
bn1 (BatchNormalization)         (128, 128, 128)         512           
pool1 (MaxPooling2D)             (32, 32, 128)           0             
_________________________________________________________________
Conv2_1 (Conv2D)                 (32, 32, 256)           295168    
bn2 (BatchNormalization)         (32, 32, 256)           1024          
pool2 (MaxPooling2D)             (8, 8, 256)             0             
_________________________________________________________________
flatten (Flatten)                (16384)                 0             
fc1 (Dense)                      (128)                   2097280   
bn3 (BatchNormalization)         (128)                   512           
dropout (Dropout)                (128)                   0             
_________________________________________________________________
fc2 (Dense)                      (64)                    8256          
bn4 (BatchNormalization)         (64)                    256           
dropout_1 (Dropout)              (64)                    0             
_________________________________________________________________
fc3 (Dense)                      (1)                     65            
=================================================================
Total params: 2,404,353
Trainable params: 2,403,201
Non-trainable params: 1,152
_________________________________________________________________
```

###5.2.5. XrayDropout2 Keras model
```
_________________________________________________________________
Layer (type)                     Output Shape            Param #   
=================================================================
Input (InputLayer)               (128, 128, 1)           0             
_________________________________________________________________
Conv1_1 (Conv2D)                 (128, 128, 64)          640           
bn1 (BatchNormalization)         (128, 128, 64)          256           
pool1 (MaxPooling2D)             (64, 64, 64)            0             
_________________________________________________________________
Conv2_1 (Conv2D)                 (64, 64, 128)           73856         
bn2 (BatchNormalization)         (64, 64, 128)           512           
pool2 (MaxPooling2D)             (32, 32, 128)           0             
_________________________________________________________________
Conv3_1 (Conv2D)                 (32, 32, 256)           295168    
bn3 (BatchNormalization)         (32, 32, 256)           1024          
pool3 (MaxPooling2D)             (16, 16, 256)           0             
_________________________________________________________________
Conv4_1 (Conv2D)                 (16, 16, 512)           1180160   
bn4 (BatchNormalization)         (16, 16, 512)           2048          
pool4 (MaxPooling2D)             (8, 8, 512)             0             
_________________________________________________________________
Conv5_1 (Conv2D)                 (8, 8, 512)             2359808   
bn5 (BatchNormalization)         (8, 8, 512)             2048          
pool5 (MaxPooling2D)             (4, 4, 512)             0             
_________________________________________________________________
flatten (Flatten)                (8192)                  0             
fc1 (Dense)                      (256)                   2097408   
bn6 (BatchNormalization)         (256)                   1024          
dropout (Dropout)                (256)                   0             
_________________________________________________________________
fc2 (Dense)                      (128)                   32896         
bn7 (BatchNormalization)         (128)                   512           
dropout_1 (Dropout)              (128)                   0             
_________________________________________________________________
fc3 (Dense)                      (64)                    8256          
bn8 (BatchNormalization)         (64)                    256           
dropout_2 (Dropout)              (64)                    0             
_________________________________________________________________
fc4 (Dense)                      (1)                     65            
=================================================================
Total params: 6,055,937
Trainable params: 6,052,097
Non-trainable params: 3,840
_________________________________________________________________
```

###5.2.6. XrayVGG16 Keras model
```
_____________________________________________________________________________________
Layer (type)                 Output Shape     Param #    Connected to                
=====================================================================================
Input (InputLayer)           (128, 128, 1)    0                  
_____________________________________________________________________________________
concatenate (Concatenate)    (128, 128, 3)    0          Input[0][0]                 
                                                         Input[0][0]                 
                                                         Input[0][0]                 
_____________________________________________________________________________________
tf_op_layer_mul               (128, 128, 3)    0         concatenate[0][0]           
Tf_op_layer_strided_slice     (28, 128, 3)     0         tf_op_layer_mul[0][0]       
tf_op_layer_BiasAdd           (128, 128, 3)    0         tf_op_layer_strided_slice[0][0]  
_____________________________________________________________________________________
vgg16 (Model)                 (4, 4, 512)      14714688  tf_op_layer_BiasAdd[0][0]            
_____________________________________________________________________________________
flatten (Flatten)             (8192)           0         vgg16[1][0]                  
fc1 (Dense)                   (1)              8193      flatten[0][0]                
=====================================================================================
Total params: 14,722,881
Trainable params: 14,722,881
Non-trainable params: 0
_____________________________________________________________________________________
```

###5.2.7. XrayMini Keras model
```
_________________________________________________________________
Layer (type)                     Output Shape        Param #   
=================================================================
Input (InputLayer)               [(128, 128, 1)]         0             
_________________________________________________________________
Conv1_1 (Conv2D)                 (128, 128, 128)         1280          
bn1 (BatchNormalization)         (128, 128, 128)         512           
pool1 (MaxPooling2D)             (32, 32, 128)           0             
_________________________________________________________________
Conv2_1 (Conv2D)                 (32, 32, 256)           295168    
bn2 (BatchNormalization)         (32, 32, 256)           1024          
pool2 (MaxPooling2D)             (8, 8, 256)             0             
_________________________________________________________________
flatten (Flatten)                (16384)                 0             
fc1 (Dense)                      (1)                     16385         
=================================================================
Total params: 314,369
Trainable params: 313,601
Non-trainable params: 768
_________________________________________________________________
```

###5.2.7. XrayOneMB Keras model
```
_________________________________________________________________
Layer (type)                   Output Shape            Param #   
=================================================================
Input (InputLayer)             (128, 128, 1)           0             
_________________________________________________________________
Conv1_1 (Conv2D)               (128, 128, 64)          640           
bn1_1 (BatchNormalization)     (128, 128, 64)          256           
Conv1_2 (Conv2D)               (128, 128, 64)          36928         
bn1_2 (BatchNormalization)     (128, 128, 64)          256           
pool1 (MaxPooling2D)           (64, 64, 64)            0             
_________________________________________________________________
Conv2_1 (Conv2D)               (64, 64, 64)            36928         
bn2_1 (BatchNormalization)     (64, 64, 64)            256           
Conv2_2 (Conv2D)               (64, 64, 64)            36928         
bn2_2 (BatchNormalization)     (64, 64, 64)            256           
pool2 (MaxPooling2D)           (32, 32, 64)            0             
_________________________________________________________________
Conv3_1 (Conv2D)               (32, 32, 128)           73856         
bn3_1 (BatchNormalization)     (32, 32, 128)           512           
Conv3_2 (SeparableConv2D)      (32, 32, 128)           17664         
bn3_2 (BatchNormalization)     (32, 32, 128)           512           
pool3 (MaxPooling2D)           (16, 16, 128)           0             
_________________________________________________________________
Conv4_1 (SeparableConv2D)      (16, 16, 128)           17664         
bn4_1 (BatchNormalization)     (16, 16, 128)           512           
_________________________________________________________________
Conv4_2 (SeparableConv2D)      (16, 16, 128)           17664         
bn4_2 (BatchNormalization)     (16, 16, 128)           512           
_________________________________________________________________
pool4 (AveragePooling2D)       (4, 4, 128)             0             
flatten (Flatten)              (2048)                  0             
_________________________________________________________________
fc1 (Dense)                    (1)                     2049          
=================================================================
Total params: 243,393
Trainable params: 241,857
Non-trainable params: 1,536
_________________________________________________________________
```