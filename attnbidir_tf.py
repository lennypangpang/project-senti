import tensorflow as tf
from dataclasses import dataclass
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    LayerNormalization,
    LSTM,
    MultiHeadAttention,
)


@dataclass
class AttnBiDirLSTMConfig:
    vocab_size: int
    embed_dim: int
    lstm_dim: int
    num_heads: int
    key_dim: int
    dropout_rate: float
    output_dim: int


class AttnBiDirLSTM(tf.keras.Model):
    """Bidirectional LSTM with multi-head cross-attention for sequence classification."""

    def __init__(self, config: AttnBiDirLSTMConfig) -> None:
        super().__init__()
        self.embed = Embedding(config.vocab_size, config.embed_dim)
        self.bidir = Bidirectional(
            LSTM(config.lstm_dim, return_sequences=True, return_state=True)
        )
        self.attention = MultiHeadAttention(
            num_heads=config.num_heads, key_dim=config.key_dim
        )
        self.norm = LayerNormalization()
        self.dropout = Dropout(config.dropout_rate)
        self.classifier = Dense(config.output_dim, activation="softmax")

    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        emb = self.embed(inputs)
        lstm_out, forward_h, _, backward_h, _ = self.bidir(emb, training=training)

        # Concat final hidden states as query: captures full sequence context
        query = tf.expand_dims(
            tf.concat([forward_h, backward_h], axis=-1), axis=1
        )

        context = self.attention(
            query=query, value=lstm_out, key=lstm_out, training=training
        )
        context = self.norm(tf.squeeze(context, axis=1))

        return self.classifier(self.dropout(context, training=training))
