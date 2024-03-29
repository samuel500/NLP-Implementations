import tensorflow as tf

from sklearn.model_selection import train_test_split

import numpy as np
import os

import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from data_utils import *
from models import *
# from eval_utils import translate


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(input_batch, target, encoder_hidden):

    loss = 0

    with tf.GradientTape() as tape:
        encoder_output, encoder_hidden = encoder(input_batch, encoder_hidden)

        decoder_hidden = encoder_hidden

        decoder_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)
        # print('dec_in', decoder_input.shape)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, target.shape[1]):
            # passing encoder_output to the decoder
            predictions, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)

            loss += loss_function(target[:, t], predictions)

            # using teacher forcing
            decoder_input = tf.expand_dims(target[:, t], 1)

    batch_loss = (loss / int(target.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def evaluate(sentence, encoder, decoder):
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

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

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



num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)


input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# Show length
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))


# print ("Input Language; index to word mapping")
# convert(inp_lang, input_tensor_train[0])
# print ()
# print ("Target Language; index to word mapping")
# convert(targ_lang, target_tensor_train[0])

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 32
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 128
units = 128
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1




if __name__=="__main__":
    print(type(input_tensor_train))
    print(len(input_tensor_train))
    raise
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    example_input_batch, example_target_batch = next(iter(dataset))
    example_input_batch.shape, example_target_batch.shape


    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

    # # sample input
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    # print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    # # print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))


    # attention_layer = AdditiveAttention(10)
    # attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

    # print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
    # print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))


    decoder = Decoder(vocab_tar_size, embedding_dim, 2*units, BATCH_SIZE)

    sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)), sample_hidden, sample_output)

    print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

    optimizer = tf.keras.optimizers.Adam()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)


    EPOCHS = 10

    for epoch in range(EPOCHS):
        start = time.time()

        encoder_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (input_batch, target)) in enumerate(dataset.take(steps_per_epoch)):
            print(input_batch)
            batch_loss = train_step(input_batch, target, encoder_hidden)
            total_loss += batch_loss
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    print(tf.train.latest_checkpoint(checkpoint_dir))
    stat = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


    translate(u'il fait tres froid ici', encoder, decoder)

    translate(u'ceci est ma vie.', encoder, decoder)

    translate(u'es tu a la maison?', encoder, decoder)

    translate(u'je suis bien content.', encoder, decoder)