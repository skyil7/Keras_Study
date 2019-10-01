from keras.datasets import mnist
from keras import models,layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# image는 0~255의 값을 갖는 unit8 타입의 2차원 텐서로 저장되어 있음

network = models.Sequential()
network.add(layers.Dense(512,activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28*28))
test_images = test_images.reshape((10000, 28*28))
# 2차원 텐서인 이미지를 1차원으로 평탄화

train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255
# 이미지의 각 픽셀의 데이터 타입을 0~1의 실수 값을 가지는 float32 자료형으로 변환

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

################################################################
# Training
################################################################

network.fit(train_images, train_labels, epochs=5, batch_size=128)

################################################################
# Inference
################################################################

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_accuracy:',test_acc)
