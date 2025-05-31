import tensorflow as tf
from keras.layers import LSTM, Dense, Embedding, Dropout, Input

class CuDNNLSTM_keras(tf.keras.Model):
    
    def __init__(self, vocab_size, embed_dim, max_len, lstm1_dim, lstm2_dim, lookup_table, dropout, output_dim = 3):
        super().__init__()
        self.max_len = max_len
         
        #model
        self.embed_layer = Embedding(

            vocab_size, 
            embed_dim, 
            weights = [lookup_table], 
            input_length = max_len, 
            trainable = False, 
            name = "Embed_Layer"
            
        )
        
        self.lstm1 = LSTM(lstm1_dim, return_sequences = True, name = "Layer_1")
        self.lstm2 = LSTM(lstm2_dim, return_sequences = False, name = "Layer_2")
        self.dense = Dense(output_dim, activation = 'softmax', name = "Dense")
        self.dropout = Dropout(dropout)

    def call(self, input_tensor):
        
        embdedding_layer = self.embed_layer(input_tensor)    
        lstm1_out = self.lstm1(embdedding_layer)
        lstm2_out = self.lstm2(lstm1_out)
        output = self.dense(lstm2_out)
        
        return output


