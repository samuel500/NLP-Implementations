import tensorflow as tf

from tensorflow.keras.layers import Dense, Layer, GRU, Embedding, Bidirectional

import numpy as np

class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.rnn = Bidirectional(GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform'))
        # self.rnn = GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        # out = self.rnn(x, initial_state=hidden)
        # print(out)
        output, state1, state2 = self.rnn(x, initial_state=hidden)

        return output, [state1, state2]

    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_size, self.enc_units))]*2
        # return tf.zeros((self.batch_size, self.enc_units))



class AdditiveAttention(Layer): 
    # https://arxiv.org/abs/1508.04025
    # http://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture08-nmt.pdf#page=78
    def __init__(self, units):
        super(AdditiveAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        query = tf.reshape(query, (values.shape[0], 1, -1))
        hidden_with_time_axis = query#tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)

        score = self.V(tf.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size #? 
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc = Dense(vocab_size)

        self.attention = AdditiveAttention(64)


    def call(self, x, hidden, encoder_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, encoder_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)

        x = self.embedding(x)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights

