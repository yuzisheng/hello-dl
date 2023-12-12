#! -*- coding:utf-8 -*-
# 一个简单的基于VAE和CNN的作诗机器人
# 来源苏神：https://kexue.fm/archives/5332

import re
import codecs
import numpy as np
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.layers import Layer
from keras.callbacks import Callback

n = 5  # 只抽取五言诗
latent_dim = 64  # 隐变量维度
hidden_dim = 64  # 隐层节点数

# 通过正则表达式找出所有五言诗
s = codecs.open("data/300_tang_poems.txt", encoding="utf-8").read()
s = re.findall(u"[\u4e00-\u9fa5]{%s}，[\u4e00-\u9fa5]{%s}。" % (n, n), s)
poems = [i[:n] + i[n + 1:len(i) - 1] for i in s if len(i) == 2 * n + 2]

# 构建字与id的相互映射
id2char = dict(enumerate(set("".join(poems))))
char2id = {j: i for i, j in id2char.items()}

# 诗歌id化
poem2id = [[char2id[j] for j in i] for i in poems]
poem2id = np.array(poem2id)


class GCNN(Layer):  # 定义GCNN层，结合残差
    def __init__(self, output_dim=None, residual=False, **kwargs):
        super(GCNN, self).__init__(**kwargs)
        self.kernel = None
        self.output_dim = output_dim
        self.residual = residual

    def build(self, input_shape):
        if self.output_dim is None:
            self.output_dim = input_shape[-1]
        self.kernel = self.add_weight(name="gcnn_kernel",
                                      shape=(3, input_shape[-1], self.output_dim * 2),
                                      initializer="glorot_uniform",
                                      trainable=True)

    def call(self, inputs, *args, **kwargs):
        _ = K.conv1d(inputs, self.kernel, padding="same")
        _ = _[:, :, :self.output_dim] * K.sigmoid(_[:, :, self.output_dim:])
        if self.residual:
            return _ + inputs
        else:
            return _


input_sentence = Input(shape=(2 * n,), dtype="int32")
input_vec = Embedding(len(char2id), hidden_dim)(input_sentence)  # id转向量
h = GCNN(residual=True)(input_vec)  # GCNN层
h = GCNN(residual=True)(h)  # GCNN层
h = GlobalAveragePooling1D()(h)  # 池化

# 算均值方差
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    _z_mean, _z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(_z_mean)[0], latent_dim), mean=0, stddev=1)
    return _z_mean + K.exp(_z_log_var / 2) * epsilon


z = Lambda(sampling)([z_mean, z_log_var])

# 定义解码层，分开定义是为了后面的重用
decoder_hidden = Dense(hidden_dim * (2 * n))
decoder_cnn = GCNN(residual=True)
decoder_dense = Dense(len(char2id), activation="softmax")

h = decoder_hidden(z)
h = Reshape((2 * n, hidden_dim))(h)
h = decoder_cnn(h)
output = decoder_dense(h)

# 建立模型
vae_poem = Model(input_sentence, output)

# xent_loss是重构loss，kl_loss是KL loss
xent_loss = K.sum(K.sparse_categorical_crossentropy(input_sentence, output), 1)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

# add_loss是新增的方法，用于更灵活地添加各种loss
vae_poem.add_loss(vae_loss)
vae_poem.compile(optimizer="adam")
vae_poem.summary()

# 重用解码层，构建单独的生成模型
decoder_input = Input(shape=(latent_dim,))
_ = decoder_hidden(decoder_input)
_ = Reshape((2 * n, hidden_dim))(_)
_ = decoder_cnn(_)
_output = decoder_dense(_)
generator = Model(decoder_input, _output)


# 利用生成模型随机生成一首诗
def gen_poem():
    r = generator.predict(np.random.randn(1, latent_dim))[0]
    r = r.argmax(axis=1)
    return "".join([id2char[i] for i in r[:n]]) + "，" + "".join([id2char[i] for i in r[n:]]) + "。"


# 回调器，方便在训练过程中输出
class Evaluate(Callback):
    def __init__(self):
        super().__init__()
        self.log = []

    def on_epoch_end(self, epoch, logs=None):
        self.log.append(gen_poem())
        print(self.log[-1])


evaluator = Evaluate()
vae_poem.fit(poem2id,
             shuffle=True,
             epochs=100,
             batch_size=64,
             callbacks=[evaluator])
vae_poem.save_weights("./poem.model")

for i in range(10):
    print("随机成诗：{}".format(gen_poem()))
print("ok")
