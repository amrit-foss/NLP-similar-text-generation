import numpy as np
import tensorflow as tf
import os
from functools import partial

tf.enable_eager_execution()

text = open("kafka.txt", 'r').read()
chars = sorted(set(text))

text_size, vocab_size = len(text), len(chars)

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = np.array(chars)


text_as_int = np.array([char_to_ix[c] for c in text])

seq_length = 100
examples_per_epoch = len(text)//seq_length

char_data_set = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_data_set.batch(seq_length+1, drop_remainder=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


data_set = sequences.map(split_input_target)

BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch//BATCH_SIZE

BUFFER_SIZE = 10000

data_set = data_set.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

embedding_dim = 256

rnn_unit = 100

rnn = partial(tf.keras.layers.GRU, recurrent_activation="sigmoid")


def build_model(vocab_size, embedding_dim, rnn_unit, batch_size):

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        rnn(rnn_unit,
            return_sequences=True,
            recurrent_initializer='glorot_uniform',
            stateful=True),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


rnn_model = build_model(vocab_size, embedding_dim, rnn_unit, BATCH_SIZE)

rnn_model.summary()


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


rnn_model.compile(optimizer=tf.train.AdamOptimizer(), loss=loss)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")


checkpoint_call_back = tf.keras.callbacks.ModelCheckpoint(checkpoint_prefix, save_weights_only=True)

EPOCHS = 5

history = rnn_model.fit(data_set.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_call_back])

tf.train.latest_checkpoint(checkpoint_dir)

trained_model = build_model(vocab_size, embedding_dim, rnn_unit, batch_size=1)

trained_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

trained_model.build(tf.TensorShape([1, None]))


def generate_text(model, start_string):
    num_generate = 1000

    input_eval = [char_to_ix[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)

        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(ix_to_char[predicted_id])

    return start_string + ''.join(text_generated)


print(generate_text(trained_model, start_string=u"ROMEO: "))
