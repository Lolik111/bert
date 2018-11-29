from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from run_classifier import InputExample, InputFeatures, DataProcessor
from enum import Enum


class RegressionProcessor(DataProcessor):
    """Processor for the ClickBait data set"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            pd.read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            pd.read_csv(os.path.join(data_dir, "train.csv")), "test")

    def get_dev_examples(self, data_dir):
        pass

    def get_labels(self):
        """See base class."""
        return ["1"]

    def _create_examples(self, df, mode):
        """Creates examples for the training and dev sets."""
        idx_tr, idx_te = next(ShuffleSplit(test_size=0.3, random_state=1234).split(df.title, df.totalViews))

        examples = []

        iterind = idx_tr if mode == "train" else idx_te

        for i in iterind:
            examples.append(
                InputExample(guid=i, text_a=df.title.values[i], label=df.totalViews.values[i]))

        return examples


class ModelFunction(object):
    def __init__(self, func, task_type):
        self.create = func
        self.task_type = task_type


class TaskType(object):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    BINARY_CLASSIFICATION = 'binary_classification'


class ModelFactory(object):

    def create_reg_model(self, output_type, head_type, dropout_val=0.9, **kwargs):

        lla = kwargs.get('last_layer_activation')
        rmsle_loss = kwargs.get('rmsle') is not None

        def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                         labels, num_labels, use_one_hot_embeddings):
            """Creates a classification model."""
            model = modeling.BertModel(
                config=bert_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings)

            if output_type == "sequence":
                output_layer = model.get_sequence_output()
            elif output_type == "pool":
                output_layer = model.get_pooled_output()
            else:
                raise NotImplementedError()

            with tf.variable_scope("loss"):
                if is_training:
                    # I.e., 0.1 dropout
                    output_layer = tf.nn.dropout(output_layer, keep_prob=dropout_val)

                if head_type == "dense" or head_type == "raw":
                    dense = tf.layers.dense(tf.layers.flatten(output_layer), 1, activation=lla,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
                    dense = tf.squeeze(dense)

                elif head_type == "2dense":
                    dense = tf.layers.dense(tf.layers.flatten(output_layer), 256, activation=tf.nn.relu,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

                    dense = tf.nn.dropout(dense, keep_prob=dropout_val)
                    dense = tf.layers.dense(dense, 1, activation=lla,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
                    dense = tf.squeeze(dense)

                elif head_type == "conv":
                    if output_type == "sequence":
                        output_layer = tf.expand_dims(output_layer, -1)
                        conv = tf.layers.conv2d(output_layer, 128, (1, 1), activation=tf.nn.relu)
                        global_avg_pool = tf.reduce_mean(conv, axis=[1, 2])
                    elif output_type == "pool":
                        output_layer = tf.expand_dims(output_layer, -1)
                        conv = tf.layers.conv1d(output_layer, 128, (1), activation=tf.nn.relu)
                        global_avg_pool = tf.reduce_mean(conv, axis=[1])
                    else:
                        raise NotImplementedError()

                    dense = tf.layers.dense(global_avg_pool, 1, activation=lla,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
                    dense = tf.squeeze(dense)

                ground_truth = tf.log1p(tf.clip_by_value(tf.cast(labels, tf.float32), 1e-8, 1e+30))
                predictions = tf.log1p(tf.clip_by_value(dense, 1e-8, 1e+30))
                msle = tf.losses.mean_squared_error(ground_truth, predictions)
                se = tf.square(ground_truth - predictions)

                if rmsle_loss == "rmsle":
                    msle = tf.sqrt(msle)
                    se = tf.sqrt(se)

                if head_type == "raw":
                    return (msle, se, dense, output_layer)

                return (msle, se, dense, predictions)

        return create_model


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, model_function):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = model_function.create(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            if model_function.task_type == TaskType.CLASSIFICATION:

                def metric_fn(per_example_loss, label_ids, logits):
                    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                    accuracy = tf.metrics.accuracy(label_ids, predictions)
                    loss = tf.metrics.mean(per_example_loss)
                    return {
                        "eval_accuracy": accuracy,
                        "eval_loss": loss,
                    }
            elif model_function.task_type == TaskType.REGRESSION:

                def metric_fn(per_example_loss, label_ids, logits):
                    ground_truth = tf.log1p(tf.clip_by_value(tf.cast(label_ids, tf.float32), 1e-8, 1e+30))
                    predictions = tf.log1p(tf.clip_by_value(logits, 1e-8, 1e+30))
                    return {
                        "eval_loss": tf.metrics.mean(per_example_loss),
                        "another_loss": tf.metrics.mean_squared_error(ground_truth, predictions)
                    }
            else:
                raise NotImplementedError()

            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=probabilities, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def input_reg_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.float32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=1000)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn
