import numpy as np
import pytest
import tensorflow as tf

from attnbidir_tf import AttnBiDirLSTM, AttnBiDirLSTMConfig


@pytest.fixture
def config() -> AttnBiDirLSTMConfig:
    return AttnBiDirLSTMConfig(
        vocab_size=1000,
        embed_dim=32,
        lstm_dim=64,
        num_heads=4,
        key_dim=16,
        dropout_rate=0.1,
        output_dim=3,
    )


@pytest.fixture
def model(config: AttnBiDirLSTMConfig) -> AttnBiDirLSTM:
    return AttnBiDirLSTM(config)


@pytest.fixture
def batch() -> tf.Tensor:
    # (batch=8, seq_len=20) integer token ids
    return tf.constant(np.random.randint(0, 1000, size=(8, 20)), dtype=tf.int32)


class TestOutputShape:
    def test_output_shape(self, model: AttnBiDirLSTM, batch: tf.Tensor) -> None:
        out = model(batch, training=False)
        assert out.shape == (8, 3), f"expected (8, 3), got {out.shape}"

    def test_single_sample_shape(self, model: AttnBiDirLSTM) -> None:
        x = tf.constant(np.random.randint(0, 1000, size=(1, 20)), dtype=tf.int32)
        out = model(x, training=False)
        assert out.shape == (1, 3)


class TestIntermediaryShapes:
    def test_embedding_shape(self, model: AttnBiDirLSTM, config: AttnBiDirLSTMConfig, batch: tf.Tensor) -> None:
        emb = model.embed(batch)
        assert emb.shape == (8, 20, config.embed_dim), f"got {emb.shape}"

    def test_bilstm_output_shape(self, model: AttnBiDirLSTM, config: AttnBiDirLSTMConfig, batch: tf.Tensor) -> None:
        emb = model.embed(batch)
        lstm_out, forward_h, _, backward_h, _ = model.bidir(emb, training=False)
        # BiLSTM doubles the hidden dim
        assert lstm_out.shape == (8, 20, config.lstm_dim * 2), f"got {lstm_out.shape}"
        assert forward_h.shape == (8, config.lstm_dim), f"forward_h: {forward_h.shape}"
        assert backward_h.shape == (8, config.lstm_dim), f"backward_h: {backward_h.shape}"

    def test_query_shape(self, model: AttnBiDirLSTM, config: AttnBiDirLSTMConfig, batch: tf.Tensor) -> None:
        emb = model.embed(batch)
        _, forward_h, _, backward_h, _ = model.bidir(emb, training=False)
        query = tf.expand_dims(tf.concat([forward_h, backward_h], axis=-1), axis=1)
        assert query.shape == (8, 1, config.lstm_dim * 2), f"got {query.shape}"

    def test_attention_output_shape(self, model: AttnBiDirLSTM, config: AttnBiDirLSTMConfig, batch: tf.Tensor) -> None:
        emb = model.embed(batch)
        lstm_out, forward_h, _, backward_h, _ = model.bidir(emb, training=False)
        query = tf.expand_dims(tf.concat([forward_h, backward_h], axis=-1), axis=1)
        context = model.attention(query=query, value=lstm_out, key=lstm_out, training=False)
        # attention output shape matches query: (batch, 1, lstm_dim*2)
        assert context.shape == (8, 1, config.lstm_dim * 2), f"got {context.shape}"


class TestOutputValidity:
    def test_probabilities_sum_to_one(self, model: AttnBiDirLSTM, batch: tf.Tensor) -> None:
        out = model(batch, training=False).numpy()
        sums = out.sum(axis=-1)
        np.testing.assert_allclose(sums, np.ones(8), atol=1e-5)

    def test_probabilities_in_range(self, model: AttnBiDirLSTM, batch: tf.Tensor) -> None:
        out = model(batch, training=False).numpy()
        assert out.min() >= 0.0, f"min prob < 0: {out.min()}"
        assert out.max() <= 1.0, f"max prob > 1: {out.max()}"

    def test_no_nans(self, model: AttnBiDirLSTM, batch: tf.Tensor) -> None:
        out = model(batch, training=False).numpy()
        assert not np.isnan(out).any(), "NaN in output"

    def test_no_infs(self, model: AttnBiDirLSTM, batch: tf.Tensor) -> None:
        out = model(batch, training=False).numpy()
        assert not np.isinf(out).any(), "Inf in output"


class TestModelCollapse:
    def test_output_not_uniform_across_batch(self, model: AttnBiDirLSTM) -> None:
        """Collapsed model outputs identical distribution for all inputs."""
        x = tf.constant(np.random.randint(0, 1000, size=(32, 20)), dtype=tf.int32)
        out = model(x, training=False).numpy()
        # Std across batch dim — near zero means collapse
        std_per_class = out.std(axis=0)
        assert std_per_class.max() > 1e-4, f"all outputs identical, max std={std_per_class.max()}"

    def test_predictions_not_all_same_class(self, model: AttnBiDirLSTM) -> None:
        """Collapsed model always predicts the same class."""
        x = tf.constant(np.random.randint(0, 1000, size=(32, 20)), dtype=tf.int32)
        out = model(x, training=False).numpy()
        predicted_classes = out.argmax(axis=-1)
        unique_classes = np.unique(predicted_classes)
        assert len(unique_classes) > 1, f"model predicts only class {unique_classes[0]} for all inputs"

    def test_gradients_nonzero(self, model: AttnBiDirLSTM, batch: tf.Tensor) -> None:
        """Zero gradients indicate dead weights or vanishing gradient collapse."""
        labels = tf.constant(np.random.randint(0, 3, size=(8,)), dtype=tf.int32)
        with tf.GradientTape() as tape:
            out = model(batch, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, out)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        all_nonzero = all(
            tf.reduce_any(g != 0.0).numpy()
            for g in grads
            if g is not None
        )
        assert all_nonzero, "some trainable weights have zero gradients"

    def test_different_inputs_produce_different_outputs(self, model: AttnBiDirLSTM) -> None:
        """Model ignores input content if outputs are identical across distinct inputs."""
        rng = np.random.default_rng(42)
        x1 = tf.constant(rng.integers(0, 1000, size=(4, 20)), dtype=tf.int32)
        x2 = tf.constant(rng.integers(0, 1000, size=(4, 20)), dtype=tf.int32)
        out1 = model(x1, training=False).numpy()
        out2 = model(x2, training=False).numpy()
        assert not np.allclose(out1, out2), "different inputs produce identical outputs"
