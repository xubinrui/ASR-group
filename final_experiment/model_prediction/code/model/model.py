'''
功能：
1.预测语音情感

最后更新时间：
2024.6.25

负责人：赵云路
'''

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Reshape, GRU, TimeDistributed
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from model_prediction.code.model.data_utils import X_test,X_train,y_test,y_train

# 定义注意力机制层
class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

# 定义模型
def create_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 5), strides=(1, 1), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 5), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Reshape((-1, x.shape[-1]*x.shape[-2]))(x)
    x = GRU(64, return_sequences=True)(x)
    x, _ = Attention(64)(x, x[:, -1, :])
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# 创建模型
model = create_model((1025, 213, 1), num_classes=6)  

#optimizer = Adam(learning_rate=0.005)
model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=32)
