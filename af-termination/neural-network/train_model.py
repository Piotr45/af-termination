"""Module for training artial termination problem."""

import argparse
import sys
import os

from typing import List, Tuple

from absl import logging

import numpy as np
import pandas as pd
import tensorflow as tf

METRICS = [
    # tf.keras.metrics.TruePositives(name='tp'),
    # tf.keras.metrics.FalsePositives(name='fp'),
    # tf.keras.metrics.TrueNegatives(name='tn'),
    # tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    # tf.keras.metrics.Precision(name='precision'),
    # tf.keras.metrics.Recall(name='recall'),
    # tf.keras.metrics.AUC(name='auc'),
    # tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]


def parse_arguments(argv: List[str]) -> argparse.Namespace:
    """The code will parse arguments from std input."""
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument(
        "--dataset",
        type=str,
        action="store",
        required=True,
        help="Directory csv file",
    )

    return arg_parser.parse_args(argv)


def split_data(df: pd.DataFrame) -> tuple:
    """This funcion will split data from dataset."""
    # read ecg data from dataset and convert it to numpy
    ecg_0 = df.iloc[1::2, :].to_numpy()
    ecg_1 = df.iloc[::2, :].to_numpy()

    # split data and labels
    train_ecg_1, labels_ecg_1 = ecg_0[:, 1:-2], ecg_0[:, -1]
    train_ecg_2, labels_ecg_2 = ecg_1[:, 1:-2], ecg_1[:, -1]

    return np.array(train_ecg_1,
                    dtype=np.float32), np.array(labels_ecg_1,
                                                dtype=np.str_), np.array(train_ecg_2,
                                                                         dtype=np.float32), np.array(labels_ecg_2,
                                                                                                     dtype=np.str_)


def create_model(input_shapes: Tuple[Tuple, Tuple]) -> tf.keras.Model():
    """This code will create model for af termination."""
    # left side learned on ECG0
    input_ecg_0 = tf.keras.Input(shape=input_shapes[0])
    ecg_0_l0 = tf.keras.layers.Dense(128, 'relu')(input_ecg_0)
    ecg_0_l1 = tf.keras.layers.Dense(64, 'relu')(ecg_0_l0)
    ecg_0_l2 = tf.keras.layers.Dense(64, 'relu')(ecg_0_l1)

    ecg_0 = tf.keras.Model(inputs=input_ecg_0, outputs=ecg_0_l2)

    # right side learned on ECG1
    input_ecg_1 = tf.keras.Input(shape=input_shapes[1])
    ecg_1_l0 = tf.keras.layers.Dense(128, 'relu')(input_ecg_1)
    ecg_1_l1 = tf.keras.layers.Dense(64, 'relu')(ecg_1_l0)
    ecg_1_l2 = tf.keras.layers.Dense(64, 'relu')(ecg_1_l1)

    ecg_1 = tf.keras.Model(inputs=input_ecg_1, outputs=ecg_1_l2)

    # concatenate models
    concatted = tf.keras.layers.Concatenate()([ecg_0.output, ecg_1.output])
    output = tf.keras.layers.Dense(3, activation='softmax')(concatted)

    # create final model
    model = tf.keras.Model(inputs=[input_ecg_0, input_ecg_1], outputs=output)

    # compile model
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=METRICS)

    return model


def convert_tflite_model(model):
    """Convert the save TF model to tflite model, then save it as .tflite flatbuffer format."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    return tflite_model


def save_tflite_model(tflite_model, save_dir, model_name):
    """Save the converted tflite model."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, model_name)
    with open(save_path, "wb") as f:
        f.write(tflite_model)
    logging.info("Tflite model saved to %s", save_dir)


def main() -> None:
    """Main script code."""
    # parse arguments
    args = parse_arguments(sys.argv[1:])
    dataset_path = args.dataset

    # read dataset
    raw_df = pd.read_csv(dataset_path)

    # split data
    train_x1, train_y1, train_x2, _ = split_data(raw_df)

    # create model
    model = create_model((train_x1.shape[1], train_x2.shape[1]))

    # prepare data for learning
    dataset1 = tf.data.Dataset.from_tensor_slices(train_x1)
    dataset2 = tf.data.Dataset.from_tensor_slices(train_x2)

    dataset = tf.data.Dataset.zip((dataset1, dataset2))
    dataset_label = tf.data.Dataset.from_tensor_slices(train_y1)
    dataset = tf.data.Dataset.zip((dataset, dataset_label)).shuffle(64, reshuffle_each_iteration=False)

    dataset = dataset.batch(64).shuffle(64, reshuffle_each_iteration=False)

    dataset_size = float(len(dataset))

    train_dataset = dataset.take(int(0.7 * dataset_size))
    test_dataset = dataset.skip(int(0.7 * dataset_size))
    test_dataset = test_dataset.take(int(0.3 * dataset_size))

    # train model
    model.fit(train_dataset, epochs=5)

    # evaluate model
    model.evaluate(test_dataset)

    # save model

    # convert model to tf lite

    # save tf lite model

    return


if __name__ == '__main__':
    main()
