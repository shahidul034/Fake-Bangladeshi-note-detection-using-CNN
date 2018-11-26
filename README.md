# Fake-Bangladeshi-note-detection-
## main.py
In this section ,we train our fake and real note using convulational neural network which implemented by keras framework.

## First We import our header file
```
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
```
## We create a Sequential class
```

classifier = Sequential()

```
## we create a seqential model.
We add 32 filter/kernel which size is 3*3 and input image size 64*64 and RGB image.For activation mode we use relu.
```
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
```
Here we have taken stride as 2, while pooling size also as 2. The max operation is applied to each depth dimension of the convolved output. As you can see, the 4*4 convolved output has become 2*2 after the max pooling operation.
```
classifier.add(MaxPooling2D(pool_size=(2, 2)))
```
Again,same code are written here.
``` 
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
```
We use total 64 filter here.
The spatial size of the output image can be calculated as( [W-F+2P]/S)+1. Here, W is the input volume size, F is the size of the filter, P is the number of padding applied and S is the number of strides. Suppose we have an input image of size 32*32*3, we apply 10 filters of size 3*3*3, with single stride and no zero padding.
W=64 , F=3 ,P=0,S=2
output image=[64-3+2*1]/2 + 1 =21

21*21*64

We data generate from our image.

```

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
								   
```
test_datagen = ImageDataGenerator(rescale=1. / 255)





classifier.add(Flatten())


classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))


classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
