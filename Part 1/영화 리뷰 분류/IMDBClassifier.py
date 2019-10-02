from keras.datasets import imdb
from keras import models, layers
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
#가장 등장 빈도가 높은 단어 10000개만 가져오기

#각 라벨은 부정을 나타내는 0과 긍정을 나타내는 1로 되어 있다.
#가장 자주 등장하는 단어 1만개로 제한했기 때문에 단어 인덱스는 9999를 넘지 않는다.

word_index = imdb.get_word_index()#단어와 정수 인덱스를 매핑한 딕션너리

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])#정수 인덱스와 단어를 매핑
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])#리뷰 디코딩 0~2는 쓰지 않음

"""
패딩을 추가하여 모든 문자열을 같은 크기로 만들고, (samples, sequence_length) 크기의 정수 텐서로 변환한다.
가장 긴 리뷰가 2494개의 단어로 이루어져 있으므로, 여기서 훈련 데이터의 크기는 (25000,2494)가 된다.

그리고 정수(단어) 시퀀스를 원-핫 인코딩하여 0과 1의 벡터로 변환한다.
즉, 10000개의 단어 중 리뷰에 사용된 단어만 1인 벡터로 만든다.
"""

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension)) #크기가 (len(sequences), dimension)이고 모든 원소가 0인 행렬 만들기
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

##########################
#신경망 모델 만들기
##########################

"""
이번에는 16개의 은닉 유닛을 가진 2개의 은닉 층을 Dense 레이어로 구축하고,
현재 리뷰의 감정을 스칼라 값의 예측으로 출력하는 세 번째 층을 둘 것이다.

은닉 층의 활성화 함수는 relu를 사용하고, 출력층에는 sigmoid 함수를 사용할 것이다.
"""

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

"""
이제 옵티마이저와 손실 함수를 설정해야 한다.

이진 분류 문제에서는 binary_crossentropy 가 적절하다.
회귀 문제에서 자주 사용하는 mean_squared_error도 좋은 선택이 될 수 있다.
"""

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

"""
검증 세트

훈련하는 동안 처음 본 데티터에 대한 모델의 정확도를 측정하기 위해서,
원본 훈련 데이터에서 10000개의 셈플을 분리해 검증 세트를 만들어야 한다.
"""

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

##########################
#모델 훈련하기
##########################

history = model.fit(partial_x_train,partial_y_train,
                    epochs=20,batch_size=512,validation_data=(x_val,y_val))

##########################
#훈련과 검증 손실 그리기
##########################

import matplotlib.pyplot as plt

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1,len(loss)+1)

plt.plot(epochs, loss, 'bo', label='Training loss')#'bo'는 파랏 점을 의미함
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

##########################
#훈련과 검증 정확도 그리기
##########################

plt.clf()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

"""
그래프로부터 epoch 4 부터 overfitting이 일어나는 것을 확인할 수 있다.
"""

##########################
#다시 해보기
##########################

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

print('학습 완료 : ', results)

print(model.predict(x_test))