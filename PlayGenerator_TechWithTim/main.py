from shutil import rmtree
from os.path import join, isdir
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

SEQ_LENGTH = 128
BATCH_SIZE = 64
EPOCHS = 50
VOCAB_SIZE = 0
EMBEDDING_DIM = 256
BUFFER_SIZE = 10000
TRAIN_SPLIT = 0.9

CHAR2IDX = {}
IDX2CHAR = []

CHECKPOINT_DIR = "checkpoint"
CHECKPOINT_CALLBACK = tf.keras.callbacks.ModelCheckpoint(
    filepath=join(CHECKPOINT_DIR, "weights_{epoch}"),
    save_weights_only=True
)

NEW_MODEL = False
CHARACTERS_TO_GENERATE = 800
SURPRISE = 1.0
START_STRING = "ROMEO:"


def main():
    train, test = preprocess_data()
    model = build_model(VOCAB_SIZE, EMBEDDING_DIM, BATCH_SIZE)

    if NEW_MODEL and isdir(CHECKPOINT_DIR):
        rmtree(CHECKPOINT_DIR)

    if not isdir(CHECKPOINT_DIR) or NEW_MODEL:
        train_model(model, train, test)

    model = build_model(VOCAB_SIZE, EMBEDDING_DIM, batch_size=1)
    model = load_model(model)
    text = generate_text(model)

    with open("result.txt", "w") as file:
        file.write(text)


def text_to_int(text):
    return np.array([CHAR2IDX[char] for char in text])


def int_to_text(ints):
    try:
        ints = ints.numpy()
    except:
        pass
    return "".join(IDX2CHAR[ints])


def preprocess_data():
    global VOCAB_SIZE, CHAR2IDX, IDX2CHAR

    with open("text.txt", "r") as file:
        text = file.read()
    del file

    vocab = sorted(set(text))
    VOCAB_SIZE = len(vocab)

    CHAR2IDX = {u: i for i, u in enumerate(vocab)}
    IDX2CHAR = np.array(vocab)

    text_as_int = text_to_int(text)
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch((SEQ_LENGTH + 1), drop_remainder=True)

    def split_input_target(chunk):
        return chunk[:-1], chunk[1:]

    dataset = sequences.map(split_input_target)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).shuffle(BUFFER_SIZE)

    train_size = int(TRAIN_SPLIT * len(dataset))
    train = dataset.take(train_size)
    test = dataset.skip(train_size)

    train = train.cache().shuffle(BUFFER_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    test = test.cache().shuffle(BUFFER_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train, test


def build_model(vocab_size, embedding_dim, batch_size):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, batch_input_shape=(batch_size, None)),
        LSTM(1024, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),
        Dropout(0.25),
        Dense(vocab_size)
    ])

    model.compile(
        optimizer="adam",
        loss=loss_function,
        metrics=["accuracy"]
    )

    return model


def train_model(model, train, test):
    model.fit(
        train,
        epochs=EPOCHS,
        callbacks=[CHECKPOINT_CALLBACK],
        validation_data=test
    )


def load_model(model: Sequential):
    model.load_weights(tf.train.latest_checkpoint(CHECKPOINT_DIR))
    model.build(tf.TensorShape([1, None]))
    return model


def generate_text(model):
    input_eval = [CHAR2IDX[char] for char in START_STRING]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    model.reset_states()
    for i in range(CHARACTERS_TO_GENERATE):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / SURPRISE
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(IDX2CHAR[predicted_id])

    return f"{START_STRING}{''.join(text_generated)}"


def loss_function(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


if __name__ == '__main__':
    main()
