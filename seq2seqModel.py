# _*_ coding:utf-8 _*_
# @Time : 2021/10/17 10:19
# @Author : xupeng
# @File : seq2seqModel.py
# @software : PyCharm

import tensorflow as tf
import getConfig

gConfig = {}
gConfig = getConfig.get_config(config_file='seq2seq.ini')
# print(gConfig)

class Encoder(tf.keras.Model):
    def __init__(self,vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder,self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True,return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        # print('x',x.shape)
        output, state = self.gru(x, initial_state=hidden)
        # print("经过GRU后的处理：")
        # print(output.shape)
        # print(output)
        # print(state.shape)
        # print(state)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz,self.enc_units))

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.w1 = tf.keras.layers.Dense(units)
        self.w2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self,query,values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)
        # print('self.w2(hidden_with_time_axis)',self.w2(hidden_with_time_axis).shape)  #(128, 1, 256)
        # print('self.w1(values)',self.w1(values).shape)  #(128, 20, 256)
        # score shape == (batch_size, max_length, hidden_size)
        # print('tanh',tf.nn.tanh(
        #     self.w1(values) + self.w2(hidden_with_time_axis)).shape)
        score = self.V(tf.nn.tanh(
            self.w1(values) + self.w2(hidden_with_time_axis)))
        # print('score', score)  #shape=(128, 20, 1)
        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(score, axis=1)  # shape=(128, 20, 1)
        # print('attention_weights',attention_weights)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values # shape=(128, 20, 256)
        # print('context_vector',context_vector)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # shape=(128, 256)
        # print('context vector',context_vector)
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder,self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self,x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)  #shape=(128, 1, 128)
        # print('x', x)
        # print('tf.expand_dims(context_vector, 1)',tf.expand_dims(context_vector, 1))
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)  #shape=(128, 1, 384)
        # print('x', x)
        output, state = self.gru(x)
        # print('output',output)  # shape=(128, 1, 256)
        # print('state',state)  # shape=(128, 256)
        output = tf.reshape(output,(-1, output.shape[2]))  #shape=(128, 256)
        x = self.fc(output)
        return x, state, attention_weights

vocab_inp_size = gConfig['enc_vocab_size']
vocab_tar_size = gConfig['dec_vocab_size']
embedding_dim = gConfig['embedding_dim']
units = gConfig['layer_size']
BATCH_SIZE = gConfig['batch_size']

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss_function(real, pred):
    # print('real',real)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    # print('mask', mask)
    loss_ = loss_object(real, pred)
    # print('loss_',loss_)  #是一个标量
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    # print('loss_', loss_)  #shape=(128,)
    # print('tf.reduce_mean(loss_)',tf.reduce_mean(loss_))
    return tf.reduce_mean(loss_) #返回是一个标量

check_point = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

#@tf.function
#传入一个batch数据，训练并输出loss
def train_step(inp, targ, targ_lang, enc_hidden):
    # print('inp')   #128 * 20
    # print(inp)
    # print('targ')  #128 * 20
    # print(targ)
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden) # enc_output 128*20*256   enc_hidden 128*256
        dec_hidden = enc_hidden
        # print('targ_lang.word_index',[targ_lang.word_index['start']])  #index是9  【9】
        dec_input = tf.expand_dims([targ_lang.word_index['start']] * BATCH_SIZE, 1)
        # print('dec_input',dec_input)
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            # print('predictions',predictions.shape) #predictions (128, 20000)
            # print('dec_hidden',dec_hidden.shape)  #dec_hidden (128, 256)
            # print('targ[:,t]',targ[:,t]) #shape=(128,)
            loss += loss_function(targ[:,t], predictions)
            dec_input = tf.expand_dims(targ[:,t], 1)
    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss






