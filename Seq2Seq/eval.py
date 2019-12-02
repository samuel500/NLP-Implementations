
import tensorflow as tf

from sklearn.model_selection import train_test_split

import numpy as np
import os

import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from data_utils import *
from models import *


def evaluate(sentence, encoder, decoder, beams=4, max_tree_width=12):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]*2

    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)


    tree = [{'dec_input': dec_input, 'dec_hidden': dec_hidden, 'ids': [], 'ps': [], 'attention_weights': []}]


    def beam_decode():
        pass


    end_hypotheses = []

    for t in range(max_length_targ):

        if len(end_hypotheses) == max_tree_width:
            break

        new_tree = []
        for branch in tree:
            # print(branch)
            predictions, dec_hidden, attention_weights = decoder(branch['dec_input'], branch['dec_hidden'], enc_out)


            top_k = tf.math.top_k(predictions[0], k=beams)
            
            attention_weights = tf.reshape(attention_weights, (-1, )).numpy()
            # attention_plot[t] = attention_weights.numpy()

            for i in range(beams):

                predicted_id = top_k[1][i]

                new_branch = {
                    'dec_input': tf.expand_dims([predicted_id], 0),
                    'dec_hidden': dec_hidden, 
                    'ids': branch['ids'] + [predicted_id.numpy()],
                    'ps': branch['ps'] + [tf.nn.softmax(predictions[0])[predicted_id].numpy()],
                    'attention_weights': branch['attention_weights'] + [attention_weights],
                }

                new_tree.append(new_branch)



            # predicted_id = tf.argmax(predictions[0]).numpy()

            # print(tf.nn.softmax(predictions[0])[predicted_id])
            # print(top_k)
            # print(tf.nn.softmax(predictions[0])[top_k[1].numpy()[1]])
            # print(targ_lang.index_word[top_k[1].numpy()[1]])
        new_tree.sort(key= lambda i: np.prod(i['ps']), reverse=True)
        new_tree = new_tree[:max_tree_width]

        ids_to_del = []
        for i, branch in enumerate(new_tree):
            if targ_lang.index_word[branch['ids'][-1]] == '<end>':
                end_hypotheses.append(branch)
                ids_to_del.append(i)

                if len(end_hypotheses) == max_tree_width:
                    break
        for i in sorted(ids_to_del, reverse=True): del new_tree[i]


        tree = new_tree

    for hypothesis in end_hypotheses:
        res = ''
        for i in hypothesis['ids']:
            res += targ_lang.index_word[i] + " "

        print(res)

    raise
    return result, sentence, attention_plot


def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def translate(sentence, encoder, decoder):
    result, sentence, attention_plot = evaluate(sentence, encoder, decoder)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))

path_to_zip = tf.keras.utils.get_file('fra-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip', extract=True)

path_to_file = os.path.dirname(path_to_zip)+"/fra-eng/fra.txt"



num_examples = 150000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)




BATCH_SIZE = 64
embedding_dim = 256
units = 512

vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

print("vocab_inp_size", vocab_inp_size)
print("vocab_tar_size", vocab_tar_size)


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)


decoder = Decoder(vocab_tar_size, embedding_dim, 2*units, BATCH_SIZE)


optimizer = tf.keras.optimizers.Adam()


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)



#print(tf.train.latest_checkpoint(checkpoint_dir))

ckpt_file = "/home/sam/DL_big_files/Seq2Seq/ckpt-5"
checkpoint.restore(ckpt_file)


while True:
    s = input("french sentence to translate to English:")

    translate(s, encoder, decoder)


# translate(u'il fait tres froid ici', encoder, decoder)

# translate(u'ceci est ma vie.', encoder, decoder)

# translate(u'es tu a la maison?', encoder, decoder)

# translate(u'je suis bien content.', encoder, decoder)