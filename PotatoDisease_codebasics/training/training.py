from os import mkdir, listdir
from os.path import join, exists, isdir
from shutil import rmtree

import tensorflow as tf
from tensorflow.keras import models, layers


_TRAINING_DIR = join("training", "PlantVillage")
_IMAGE_SIZE = 256
_BATCH_SIZE = 12
_CHANNELS = 3
_EPOCHS = 50

_TRAIN_SIZE = 0.75
_VALIDATION_SIZE = 0.1
_CLASSES = 3

_MAJOR_VERSION = 1.0


def main():
    dataset = get_dataset()
    training_dataset, validation_dataset, test_dataset = preprocess_data(dataset, _TRAIN_SIZE, _VALIDATION_SIZE,
                                                                         True, 5000)
    model = get_model()
    model, history = train(model, training_dataset, validation_dataset)
    model.evaluate(test_dataset)

    save_model(model)


def get_dataset():
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory=_TRAINING_DIR,
        shuffle=True,
        image_size=(_IMAGE_SIZE, _IMAGE_SIZE),
        batch_size=_BATCH_SIZE
    )


def preprocess_data(dataset, train_split, validation_split, shuffle, shuffle_size):
    dataset_size = len(dataset)

    if shuffle:
        dataset = dataset.shuffle(shuffle_size, seed=shuffle_size)

    training_size = int(train_split * dataset_size)
    validation_size = int(validation_split * dataset_size)

    training_dataset = dataset.take(training_size)
    validation_dataset = dataset.skip(training_size).take(validation_size)
    test_dataset = dataset.skip(training_size + validation_size)

    training_dataset = training_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return training_dataset, validation_dataset, test_dataset


def get_model():
    div = 1
    input_shape = (_BATCH_SIZE, int(_IMAGE_SIZE * div), int(_IMAGE_SIZE * div), _CHANNELS)

    resize_and_rescale = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(int(_IMAGE_SIZE * div), int(_IMAGE_SIZE * div)),
        layers.experimental.preprocessing.Rescaling(1.0 / 255),
    ])

    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    model = models.Sequential([
        resize_and_rescale,
        data_augmentation,
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2), strides=(1, 1)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(1, 1)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(1, 1)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(_CLASSES, activation='softmax'),
    ])

    model.build(input_shape=input_shape)

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    return model


def train(model, training_dataset, validation_dataset):

    history = model.fit(
        training_dataset,
        epochs=_EPOCHS,
        batch_size=_BATCH_SIZE,
        verbose=1,
        validation_data=validation_dataset,
    )

    return model, history


def save_model(model):
    base_model_dir = "saved_models"

    if not isdir(base_model_dir):
        mkdir(base_model_dir)

    model_version = _MAJOR_VERSION + (len(listdir(base_model_dir)) * 0.1)
    model_dir = join(base_model_dir, f"v{model_version}.h5")

    if exists(model_dir):
        rmtree(model_dir)

    model.save(model_dir)


if __name__ == '__main__':
    with tf.device("/GPU:0"):
        main()
