import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.layers import Attention

class AttnBiDirLSTM(tf.keras.Model):
    
    def __init__(self, vocab_size, embed_dim, max_len, lstm1_dim, dropout, output_dim = 3):
        super().__init__()
        #param
        self.max_len = max_len
         
        #model
        self.embed = Embedding(vocab_size, embed_dim)
        self.bidir = Bidirectional(LSTM(lstm1_dim, return_sequences = True, return_state = True))
        self.dropout_layer = Dropout(dropout)
        self.attention_layer = Attention()
        self.output_layer = Dense(output_dim, activation = 'softmax')

    def call(self, input_tensor, training = True):

        emb = self.embed(input_tensor)
        
        lstm_out, forward_h, forward_c, backward_h, backward_c = self.bidir(emb, training = training)
        
        #k, v pairs for attn
        key = lstm_out
        value = lstm_out
        
        #query vector
        query = tf.concat([forward_h, backward_h], axis = -1)
        query = tf.expand_dims(query, 1)
        
        #comp attn
        context_vector = self.attention_layer(inputs = [query, value])
        context_vector = tf.squeeze(context_vector, axis = 1)
        
        output = self.dropout_layer(context_vector, training = training)
        
        return self.output_layer(output) 
        
    






