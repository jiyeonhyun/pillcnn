# Pre_processing


```python
## Check out working directory
%pwd
```




    'C:\\Users\\stu15'




```python
## Import module 

import loader_hyug
import os 
import re
import cv2
import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
```

## Resize


```python
## Resize & create file by order:asc
# for modeling

path_raw = "D:\\LSS_project\\Data\\raw"
path_re = "D:\\LSS_project\\Data\\resize"

loader_hyug.resize(path_raw, path_re, height=128, width=128)  #default = 128*128
```


    
![png](output_4_0.png)
    


    img shape (128, 128, 3)
    

## Load X : Features


```python
## Load X : Features

path_re = "D:\\LSS_project\\Data\\resize"
resize = loader_hyug.image_load(path_re)
```


```python
## Check out first img(resized)

img = resize[0, : ]
plt.figure()
plt.imshow(img)
print(resize.shape)  
```

    (6000, 128, 128, 3)
    


    
![png](output_7_1.png)
    



```python
## Make X (feature)

X = resize
X.shape  
```




    (6000, 128, 128, 3)



## Create Label

### Create Csv file


```python
## Make csv file in path_re

path_re = "D:\\LSS_project\\Data\\re.csv"
loader_hyug.csv_maker_84(path_re, k1=10, k2=600)

```

### Load y : label


```python
path_re = "D:\\LSS_project\\Data\\re.csv"
y = loader_hyug.label_load(path_re,label_cnt=10)  
y.shape
```




    (6000, 10)



## Scailing


```python
X = X.astype('float')
X = X/255
X.shape  
```




    (6000, 128, 128, 3)



## Check X, y


```python
## Confirm X, y
print(X.shape)  
print(y.shape, end='\n\n\n')  

# print("#####Check out : X#####")
# print(X, end='\n\n\n')
# print("#####Check out : y#####")
# print(y)
```

    (6000, 128, 128, 3)
    (6000, 10)
    
    
    

  

   

# Modeling


```python
## module import
import tensorflow as tf # tensorflow 2.0
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Activation
from tensorflow.keras.layers import Dropout, BatchNormalization, Dense
from tensorflow.keras.optimizers import Adam


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


## 전이학습
from tensorflow.keras.applications.vgg16 import VGG16                        
from tensorflow.keras.applications.vgg19 import VGG19
```


```python
## Check tensorFlow version

print (tf.__version__)
```

    2.4.1
    


```python
# 훈련/테스트 데이터를 0.7/0.3의 비율로 분리합니다.
x_train, x_val, y_train, y_val = train_test_split(X, y, 
                                                test_size = 0.3, 
                                                random_state = 333)

# Checkout
print(x_train.shape)
print(x_val.shape)  
print(y_train.shape)
print(y_val.shape)  
```

    (4200, 128, 128, 3)
    (1800, 128, 128, 3)
    (4200, 10)
    (1800, 10)
    


```python
## VGG16 options

vgg16 = VGG16(weights = 'imagenet', include_top = False, input_shape = (128, 128, 3))
vgg16.summary()
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    58892288/58889256 [==============================] - 4s 0us/step
    Model: "vgg16"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 128, 128, 3)]     0         
    _________________________________________________________________
    block1_conv1 (Conv2D)        (None, 128, 128, 64)      1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 128, 128, 64)      36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 64, 64, 64)        0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 64, 64, 128)       73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 64, 64, 128)       147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 32, 32, 128)       0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 32, 32, 256)       295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 32, 32, 256)       590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 32, 32, 256)       590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 16, 16, 256)       0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 16, 16, 512)       1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 16, 16, 512)       2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 16, 16, 512)       2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 8, 8, 512)         0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 8, 8, 512)         2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 8, 8, 512)         2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 8, 8, 512)         2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
    =================================================================
    Total params: 14,714,688
    Trainable params: 14,714,688
    Non-trainable params: 0
    _________________________________________________________________
    


```python
## 가중치 초기값 : imagenet
# layer.trainable=True : 동결 해제 (default)
# layer.trainable=False : 동결 (option)

for layer in vgg16.layers[:15]:
    layer.trainable = False
```


```python
vgg16.summary()
```

    Model: "vgg16"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 128, 128, 3)]     0         
    _________________________________________________________________
    block1_conv1 (Conv2D)        (None, 128, 128, 64)      1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 128, 128, 64)      36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 64, 64, 64)        0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 64, 64, 128)       73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 64, 64, 128)       147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 32, 32, 128)       0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 32, 32, 256)       295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 32, 32, 256)       590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 32, 32, 256)       590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 16, 16, 256)       0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 16, 16, 512)       1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 16, 16, 512)       2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 16, 16, 512)       2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 8, 8, 512)         0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 8, 8, 512)         2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 8, 8, 512)         2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 8, 8, 512)         2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
    =================================================================
    Total params: 14,714,688
    Trainable params: 7,079,424
    Non-trainable params: 7,635,264
    _________________________________________________________________
    


```python
# 신경망 객체 생성
model = Sequential()

# stacking vgg16
model.add(vgg16)

# Reshape : Flatten 
model.add(Flatten())

# 완전연결계층1
model.add(Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.5)))  #default node num = 4096
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))  #traditional range : 0.2~0.5

# 완전연결계층2
model.add(Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.5)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

# 출력층(softmax)
model.add(Dense(10))  # class : 10
model.add(Activation('softmax'))

# Check out model 
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    vgg16 (Functional)           (None, 4, 4, 512)         14714688  
    _________________________________________________________________
    flatten (Flatten)            (None, 8192)              0         
    _________________________________________________________________
    dense (Dense)                (None, 256)               2097408   
    _________________________________________________________________
    batch_normalization (BatchNo (None, 256)               1024      
    _________________________________________________________________
    activation (Activation)      (None, 256)               0         
    _________________________________________________________________
    dropout (Dropout)            (None, 256)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 256)               65792     
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 256)               1024      
    _________________________________________________________________
    activation_1 (Activation)    (None, 256)               0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 256)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                2570      
    _________________________________________________________________
    activation_2 (Activation)    (None, 10)                0         
    =================================================================
    Total params: 16,882,506
    Trainable params: 9,246,218
    Non-trainable params: 7,636,288
    _________________________________________________________________
    


```python
# model.compile(loss='categorical_crossentropy', 
#               optimizer=Adam(lr = 0.0001), 
#               metrics=['accuracy'])

model.compile(loss='categorical_crossentropy', 
              optimizer="adam", 
              metrics=['accuracy'])
```


```python
epochs = 10
batch_size=50

hist = model.fit(x_train, y_train, 
                 validation_data=(x_val, y_val), 
                 epochs=epochs, 
                 batch_size=batch_size)
```


```python
## Check out error
scores3 = model.evaluate(ind_X, ind_y, verbose=0)

print("Error : %.2f%%" % (100-scores3[1]*100))
```


```python
# list all data in history
print(hist.history.keys())
```


```python
## Visualization : Accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])

plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

## Visualization : Loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'val'], loc='upper right')
plt.show()
```


```python
## Check out accuracy

scores = model.evaluate(x_train, y_train, verbose=0)
scores2 = model.evaluate(x_val, y_val, verbose=0)

print("Vgg16 train Error : %.2f%%" % (100-scores[1]*100))
print("Vgg16 val Error : %.2f%%" % (100-scores2[1]*100))
```


```python
## Save model : .h5(Hdf5 type file)

save_path = "d:\\medicine_vgg16.h5"
save_model(model, save_path)  
```

 

 
    

