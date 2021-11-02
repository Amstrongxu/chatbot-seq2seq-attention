# _*_ coding:utf-8 _*_
# @Time : 2021/10/17 13:20
# @Author : xupeng
# @File : execute.py
# @software : PyCharm
import os
import sys
import time
import tensorflow as tf
import seq2seqModel
import getConfig
import io

gConfig = {}
gConfig = getConfig.get_config(config_file='seq2seq.ini')
# print(gConfig)
vocab_inp_size = gConfig['enc_vocab_size']
vocab_tar_size = gConfig['dec_vocab_size']
embedding_dim = gConfig['embedding_dim']
units = gConfig['layer_size']
BATCH_SIZE = gConfig['batch_size']
max_length_inp, max_length_tar = 20, 20

def preprocess_sentence(w):
    w = 'start ' + w + ' end'
    return w

def create_dataset(path, num_examples):
    lines = io.open(path, encoding='utf-8').read().strip().split('\n')
    # print("cratedataset- lines",lines)
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)

def max_length(tensor):
    return max(len(t) for t in tensor)

def read_data(path, num_examples):
    input_lang, target_lang = create_dataset(path, num_examples)
    # print('input_lang',input_lang[:5])
    # print('target_lang',target_lang[:5])
    input_tensor, input_token = tokenize(input_lang)
    # print('input_tensor',input_tensor)
    target_tensor, target_token = tokenize(target_lang)
    return input_tensor, input_token, target_tensor, target_token

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=gConfig['enc_vocab_size'], oov_token=3)
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_length_inp, padding='post')
    return tensor, lang_tokenizer

input_tensor, input_token, target_tensor, target_token = read_data(gConfig['seq_data'], gConfig['max_train_data_size'])
# print(len(input_tensor))
# print(len(target_tensor))
def train():
    print('Preparing data in %s' % gConfig['train_data'])
    steps_per_epoch = len(input_tensor) // gConfig['batch_size']  #代表一轮有多少个batch
    print(steps_per_epoch)
    enc_hidden = seq2seqModel.encoder.initialize_hidden_state()   #128 * 256
    # print("enc_hidden", enc_hidden.shape)

    checkpoint_dir = gConfig['model_data']
    if not os.path.exists('./' + checkpoint_dir):
        os.makedirs('./' + checkpoint_dir)
    # print('checkpoint_dir',checkpoint_dir)
    ckpt = tf.io.gfile.listdir(checkpoint_dir) #返回路径下包含的条目列表
    if ckpt: #列表不为空，说明之前保存了训练好的模型
        print("reload pretrained model")
        seq2seqModel.check_point.restore(tf.train.latest_checkpoint(checkpoint_dir)) #加载模型
    BUFFER_SIZE = len(input_tensor)
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    # print('dataset', dataset)
    # print('dataset', len(dataset))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)  #变成长度为390的列表每个元素就是一个batch
    # print('dataset.take(steps_per_epoch)',dataset.take(steps_per_epoch)) #shapes: ((128, 20), (128, 20))
    checkpoint_dir = gConfig['model_data']
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    # print("checkpoint_prefix",checkpoint_prefix)
    start_time = time.time()
    while True:
        start_time_epoch = time.time()
        total_loss = 0
        for (batch,(inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            # print('batch',batch)
            # print('inp',inp.shape)  #128 * 20
            # print('targ',targ.shape)  #128 * 20
            batch_loss = seq2seqModel.train_step(inp, targ, target_token, enc_hidden) #传入一个batch的数据
            total_loss += batch_loss
            print(batch_loss.numpy())
        step_time_epoch = (time.time() - start_time_epoch) / steps_per_epoch
        step_loss = total_loss / steps_per_epoch
        current_steps = +steps_per_epoch
        # print('current_steps',current_steps)
        step_time_total = (time.time() - start_time) / current_steps
        print('训练总步数: {} 每步耗时: {}  最新每步耗时: {} 最新每步loss {:.4f}'.format(current_steps, step_time_total, step_time_epoch,
                                                                      step_loss.numpy()))
        seq2seqModel.check_point.save(file_prefix=checkpoint_prefix)
        sys.stdout.flush()

def predict(sentence):
    checkpoint_dir = gConfig['model_data']
    seq2seqModel.check_point.restore(tf.train.latest_checkpoint(checkpoint_dir))
    sentence = preprocess_sentence(sentence)
    print('sentence',sentence)
    inputs = [input_token.word_index.get(i, 3) for i in sentence.split(' ')]
    print('inputs', inputs)
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    print('inputs', inputs)
    inputs = tf.convert_to_tensor(inputs)
    print('inputs', inputs)
    result = ''
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = seq2seqModel.encoder(inputs, hidden)
    print('enc_out',enc_out)  #shape=(1, 20, 256)
    print('enc_hidden', enc_hidden)  #shape=(1, 256)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_token.word_index['start']], 0)
    print('dec_input',dec_input)
    for t in range(max_length_tar):
        preditions, dec_hidden, attention_weights = seq2seqModel.decoder(dec_input, dec_hidden, enc_out)
        print('preditions',preditions)
        predicted_id = tf.argmax(preditions[0]).numpy()
        if target_token.index_word[predicted_id] == 'end':
            break
        result += target_token.index_word[predicted_id] + ' '
        dec_input = tf.expand_dims([predicted_id], 0)
    print('result',result)
    return result

if __name__ == "__main__":
    print('sys.argv',sys.argv)
    if len(sys.argv) - 1:
        gConfig = getConfig.get_config(sys.argv[1])
    else:
        gConfig = getConfig.get_config()
    print('\n>>Mode: %s\n' %(gConfig['mode']))
    if gConfig['mode'] == 'train':
        train()
    elif gConfig['mode'] == 'serve':
        print('Serve Usage: >> python3 app.py')








