import tensorflow as tf
from dataclasses import dataclass
from tensorflow.keras.layers import Dense, Dropout, Embedding, LayerNormalization, LSTM


@dataclass
class StackedLSTMConfig:
    vocab_size: int
    embed_dim: int
    lstm1_dim: int
    lstm2_dim: int
    lookup_table: object
    dropout_rate: float
    output_dim: int


class StackedLSTM(tf.keras.Model):
    """Two-layer stacked LSTM with pretrained embeddings for sequence classification."""

    def __init__(self, config: StackedLSTMConfig) -> None:
        super().__init__()
        self.embed = Embedding(
            config.vocab_size,
            config.embed_dim,
            weights=[config.lookup_table],
            trainable=False,
        )
        self.lstm1 = LSTM(config.lstm1_dim, return_sequences=True)
        self.norm = LayerNormalization()
        self.dropout = Dropout(config.dropout_rate)
        self.lstm2 = LSTM(config.lstm2_dim, return_sequences=False)
        self.classifier = Dense(config.output_dim, activation="softmax")

    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        x = self.embed(inputs)
        x = self.lstm1(x, training=training)
        x = self.norm(x)
        x = self.dropout(x, training=training)
        x = self.lstm2(x, training=training)
        return self.classifier(x)
