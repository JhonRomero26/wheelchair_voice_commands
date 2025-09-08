# coding=utf-8
# Copyright 2023 The Google Research Authors.
# Modifications copyright 2024 Kevin Ahrendt.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import contextlib

from absl import logging

import numpy as np
import tensorflow as tf

from tensorflow.python.util import tf_decorator
from tensorflow.keras.utils import to_categorical


@contextlib.contextmanager
def swap_attribute(obj, attr, temp_value):
    """Temporarily swap an attribute of an object."""
    original_value = getattr(obj, attr)
    setattr(obj, attr, temp_value)

    try:
        yield
    finally:
        setattr(obj, attr, original_value)

def preprocess_data(set_data, num_classes):
        batch_x, batch_y, _ = set_data
        batch_x = np.array(batch_x).astype(np.float32)
        batch_y = to_categorical(batch_y, num_classes=num_classes)
        return batch_x, batch_y

def validate_nonstreaming(config, data_processor, model, test_set):
    num_classes = config["num_classes"]
    val_set = data_processor.get_data(
        test_set,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy="truncate_start",
    )
    x_val, y_val = preprocess_data(val_set, num_classes)

    model.reset_metrics()

    result = model.evaluate(
        x_val,
        y_val,
        batch_size=1024,
        return_dict=True,
        verbose=0,
    )

    return result


def train(model, config, data_processor):
    # Assign default training settings if not set in the configuration yaml
    if not (training_steps_list := config.get("training_steps")):
        training_steps_list = [20000]
    if not (learning_rates_list := config.get("learning_rates")):
        learning_rates_list = [0.001]
    if not (mix_up_prob_list := config.get("mix_up_augmentation_prob")):
        mix_up_prob_list = [0.0]
    if not (freq_mix_prob_list := config.get("freq_mix_augmentation_prob")):
        freq_mix_prob_list = [0.0]
    if not (time_mask_max_size_list := config.get("time_mask_max_size")):
        time_mask_max_size_list = [5]
    if not (time_mask_count_list := config.get("time_mask_count")):
        time_mask_count_list = [2]
    if not (freq_mask_max_size_list := config.get("freq_mask_max_size")):
        freq_mask_max_size_list = [5]
    if not (freq_mask_count_list := config.get("freq_mask_count")):
        freq_mask_count_list = [2]
    
    num_classes = config["num_classes"]
    batch_size = config["batch_size"]
    train_dir = config["train_dir"]

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.TopKCategoricalAccuracy(name="top3", k=3),
    ]
    optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    checkpoint_dir = os.path.join(train_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Configure TensorBoard summaries
    summary_writer = tf.summary.create_file_writer(
        os.path.join(config["summaries_dir"])
    )

    for i in range(len(training_steps_list)):
        learning_rate = learning_rates_list[i]
        mix_up_prob = mix_up_prob_list[i]
        freq_mix_prob = freq_mix_prob_list[i]
        time_mask_max_size = time_mask_max_size_list[i]
        time_mask_count = time_mask_count_list[i]
        freq_mask_max_size = freq_mask_max_size_list[i]
        freq_mask_count = freq_mask_count_list[i]
        break

    model.optimizer.learning_rate.assign(learning_rate)

    augmentation_policy = {
        "mix_up_prob": mix_up_prob,
        "freq_mix_prob": freq_mix_prob,
        "time_mask_max_size": time_mask_max_size,
        "time_mask_count": time_mask_count,
        "freq_mask_max_size": freq_mask_max_size,
        "freq_mask_count": freq_mask_count,
    }

    train_set = data_processor.get_data(
        "training",
        batch_size=batch_size,
        features_length=config["spectrogram_length"],
        truncation_strategy="truncate_start",
        augmentation_policy=augmentation_policy,
    )

    x_train, y_train = preprocess_data(train_set, num_classes)

    epochs = config.get("epochs", 5)
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        num_batches = len(x_train) // batch_size

        # Loop de batches
        for i in range(num_batches):
            bx = x_train[i * batch_size : (i + 1) * batch_size]
            by = y_train[i * batch_size : (i + 1) * batch_size]
            result = model.train_on_batch(bx, by, return_dict=True)
            print(
                f"Batch {i+1}/{num_batches} - "
                + " - ".join([f"{k}: {v:.4f}" for k, v in result.items()]),
                end="\r",
            )

        # Validación
        val_result = validate_nonstreaming(config, data_processor, model, "validation")
        print("\nValidation:", " - ".join([f"{k}: {v:.4f}" for k, v in val_result.items()]))

        if epoch == 1 or val_result["accuracy"] >= best_val_acc:
            best_val_acc = val_result["accuracy"]
            model.save_weights(os.path.join(train_dir, "best_weights.weights.h5"))

        # Guardar weights
        model.save_weights(os.path.join(train_dir, "last_weights.weights.h5"))
        model.save_weights(os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.weights.h5"))

        # Guardar en TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar("train/loss", result["loss"], step=epoch)
            tf.summary.scalar("train/accuracy", result["accuracy"], step=epoch)
            tf.summary.scalar("val/loss", val_result["loss"], step=epoch)
            tf.summary.scalar("val/accuracy", val_result["accuracy"], step=epoch)

    final_path = os.path.join(train_dir, "final_model")
    model.export(final_path)
    print(f"\nModelo final guardado en {final_path}")


        # train_ground_truth = np.array(train_ground_truth).astype(np.int32).reshape(-1,)
        # train_ground_truth = to_categorical(train_ground_truth, num_classes=num_classes)
        # print("Batch shapes:", train_fingerprints.shape, train_ground_truth.shape)


        # class_weights = {idx: positive_class_weight if idx > 0 else negative_class_weight for idx in range(num_classes)}
        # train_sample_weights = np.array([class_weights[np.argmax(i)] for i in train_ground_truth])
            
        # result = model.train_on_batch(
        #     train_fingerprints,
        #     train_ground_truth,
        #     sample_weight=train_sample_weights,
        # )

        # Mapear por nombre para ser robustos a cambios
        # if isinstance(result, (list, tuple)):
        #     result_dict = dict(zip(model.metrics_names, result))
        # else:
        #     # algunos TF devuelven solo un escalar (loss)
        #     result_dict = {"loss": float(result)}

        # loss_val = result_dict.get("loss", None)
        # accuracy_val = result_dict.get("accuracy", result_dict.get("sparse_categorical_accuracy", 0.0))
        # top1_val = result_dict.get("top1", 0.0)
        # top3_val = result_dict.get("top3", 0.0)
        # auc_val = result_dict.get("auc", 0.0)

        # # Print the running statistics in the current validation epoch
        # print(
        #     "Step {:d}: loss={:.4f}; acc={:.4f}; top1={:.4f}; top3={:.4f}; auc={:.4f}\r".format(
        #         training_step, loss_val, accuracy_val, top1_val, top3_val, auc_val
        #     ),
        #     end="",
        # )

        # is_last_step = training_step == training_steps_max
        # if (training_step % config["eval_step_interval"]) == 0 or is_last_step:
        #     print(
        #         "Step {:d}: loss={:.4f}; acc={:.4f}; top1={:.4f}; top3={:.4f}; auc={:.4f}\r".format(
        #             training_step, loss_val, accuracy_val, top1_val, top3_val, auc_val
        #         ),
        #         end="",
        #     )

        #     with train_writer.as_default():
        #         tf.summary.scalar("loss", result[9], step=training_step)
        #         tf.summary.scalar("accuracy", result[1], step=training_step)
        #         tf.summary.scalar("recall", result[2], step=training_step)
        #         tf.summary.scalar("precision", result[3], step=training_step)
        #         tf.summary.scalar("auc", result[8], step=training_step)
        #         train_writer.flush()

            # model.save_weights(
            #     os.path.join(config["train_dir"], "last_weights.weights.h5")
            # )

            # nonstreaming_metrics = validate_nonstreaming(
            #     config, data_processor, model, "validation"
            # )
            # model.reset_metrics()  # reset metrics for next validation epoch of training
            # logging.info(
            #     "Step %d (nonstreaming): Validation: recall at no faph = %.3f with cutoff %.2f, accuracy = %.2f%%, recall = %.2f%%, precision = %.2f%%, ambient false positives = %d, estimated false positives per hour = %.5f, loss = %.5f, auc = %.5f, average viable recall = %.9f",
            #     *(
            #         training_step,
            #         nonstreaming_metrics["recall_at_no_faph"] * 100,
            #         nonstreaming_metrics["cutoff_for_no_faph"],
            #         nonstreaming_metrics["accuracy"] * 100,
            #         nonstreaming_metrics["recall"] * 100,
            #         nonstreaming_metrics["precision"] * 100,
            #         nonstreaming_metrics["ambient_false_positives"],
            #         nonstreaming_metrics["ambient_false_positives_per_hour"],
            #         nonstreaming_metrics["loss"],
            #         nonstreaming_metrics["auc"],
            #         nonstreaming_metrics["average_viable_recall"],
            #     ),
            # )

    #         with validation_writer.as_default():
    #             tf.summary.scalar(
    #                 "loss", nonstreaming_metrics["loss"], step=training_step
    #             )
    #             tf.summary.scalar(
    #                 "accuracy", nonstreaming_metrics["accuracy"], step=training_step
    #             )
    #             tf.summary.scalar(
    #                 "recall", nonstreaming_metrics["recall"], step=training_step
    #             )
    #             tf.summary.scalar(
    #                 "precision", nonstreaming_metrics["precision"], step=training_step
    #             )
    #             tf.summary.scalar(
    #                 "recall_at_no_faph",
    #                 nonstreaming_metrics["recall_at_no_faph"],
    #                 step=training_step,
    #             )
    #             tf.summary.scalar(
    #                 "auc",
    #                 nonstreaming_metrics["auc"],
    #                 step=training_step,
    #             )
    #             tf.summary.scalar(
    #                 "average_viable_recall",
    #                 nonstreaming_metrics["average_viable_recall"],
    #                 step=training_step,
    #             )
    #             validation_writer.flush()

    #         os.makedirs(os.path.join(config["train_dir"], "train"), exist_ok=True)

    #         model.save_weights(
    #             os.path.join(
    #                 config["train_dir"],
    #                 "train",
    #                 f"{int(best_minimization_quantity * 10000)}_weights_{training_step}.weights.h5",
    #             )
    #         )

    #         current_minimization_quantity = 0.0
    #         if config["minimization_metric"] is not None:
    #             current_minimization_quantity = nonstreaming_metrics[
    #                 config["minimization_metric"]
    #             ]
    #         current_maximization_quantity = nonstreaming_metrics[
    #             config["maximization_metric"]
    #         ]
    #         current_no_faph_cutoff = nonstreaming_metrics["cutoff_for_no_faph"]

    #         # Save model weights if this is a new best model
    #         if (
    #             (
    #                 (
    #                     current_minimization_quantity <= config["target_minimization"]
    #                 )  # achieved target false positive rate
    #                 and (
    #                     (
    #                         current_maximization_quantity > best_maximization_quantity
    #                     )  # either accuracy improved
    #                     or (
    #                         best_minimization_quantity > config["target_minimization"]
    #                     )  # or this is the first time we met the target
    #                 )
    #             )
    #             or (
    #                 (
    #                     current_minimization_quantity > config["target_minimization"]
    #                 )  # we haven't achieved our target
    #                 and (
    #                     current_minimization_quantity < best_minimization_quantity
    #                 )  # but we have decreased since the previous best
    #             )
    #             or (
    #                 (
    #                     current_minimization_quantity == best_minimization_quantity
    #                 )  # we tied a previous best
    #                 and (
    #                     current_maximization_quantity > best_maximization_quantity
    #                 )  # and we increased our accuracy
    #             )
    #         ):
    #             best_minimization_quantity = current_minimization_quantity
    #             best_maximization_quantity = current_maximization_quantity
    #             best_no_faph_cutoff = current_no_faph_cutoff

    #             # overwrite the best model weights
    #             model.save_weights(
    #                 os.path.join(config["train_dir"], "best_weights.weights.h5")
    #             )
    #             checkpoint.save(file_prefix=checkpoint_prefix)

    #         logging.info(
    #             "So far the best minimization quantity is %.3f with best maximization quantity of %.5f%%; no faph cutoff is %.2f",
    #             best_minimization_quantity,
    #             (best_maximization_quantity * 100),
    #             best_no_faph_cutoff,
    #         )

    # # Save checkpoint after training
    # checkpoint.save(file_prefix=checkpoint_prefix)
    # model.save_weights(os.path.join(config["train_dir"], "last_weights.weights.h5"))
