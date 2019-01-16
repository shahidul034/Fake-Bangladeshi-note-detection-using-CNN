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

21 * 21 * 64

We data generate for training our system.

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
```
# Flow diagram
![alt text](https://github.com/shahidul034/Fake-Bangladeshi-note-detection-/blob/master/pic/flow%20diagram2.jpg)

![alt text](https://github.com/shahidul034/Fake-Bangladeshi-note-detection-/blob/master/pic/take%20flow%20diagram.jpg)

![alt text](https://github.com/shahidul034/Fake-Bangladeshi-note-detection-/blob/master/pic/test.jpg)

![alt text](https://github.com/shahidul034/Fake-Bangladeshi-note-detection-/blob/master/pic/test2.jpg)

![alt text](https://github.com/shahidul034/Fake-Bangladeshi-note-detection-/blob/master/pic/train.jpg)

# Conclusion
1.We have tried to develop a model that can verify a detect the fake currency of Bangladesh.

2.Our designed model is applicable to only 100 tk for lacking dataset because it is difficult for us to collect 500 and 1000 tk
# Limitation
Here, we do not have real dataset we collect our own dataset  

We just collect 100 tk because 500 tk is not available and we need up to 120 image for this.
Because of dataset  it affects of our accuracy
# Future Work
If we increase our dataset,it helps us to increase our accuracy

If anyone collect 500 and 1000 tk,he can use our model to train and test their dataset.
# References
[1] Fake Currency Detection Using Image Processing and OtherStandard MethodsD.Alekhya , G.DeviSuryaPrabha , G.VenkataDurgaRao 
[2] Image Processing Based Feature Extraction ofBangladeshi BanknotesZahid Ahmed, Sabina Yasmin, Md Nahidul Islam, Raihan Uddin Ahmed 
[3] Currency Recognition System Using Image ProcessingS. M. Saifullah1, AnikaRahmanAnanna2, Md. Shakhawat Hossain3, Md. JaouadHossain4,Md. SaniatRahman Zishan5 [4] Image-Based Approach for the Detection ofCounterfeit Banknotes of BangladeshMohammad Shorif Uddin, Pronaya Prosun Das, Md. Shamim Ahmed Roney 
[4] Detection of Fake Indian Currency ‘’ Gouri Sanjay Tele’’
[5]https://www.analyticsvidhya.com/blog/2017/06/architecture-of-convolutional-neural-networks-simplified-demystified/?fbclid=IwAR2s_ZCYE7pUPtVtiVkHWnhbkbHJuJtDMsdZJ80S3ZjT5f4g4xxuQ3YhwRU
[6] https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8
[7] https://keras.io/models/sequential/
[8] https://www.coursera.org/specializations/deep-learning


