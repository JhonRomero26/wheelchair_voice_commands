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

"""Test utility functions for accuracy evaluation."""

import os

import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import yaml

from absl import logging
from typing import List
from speech_commands.inference import Model
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize


def get_classes(config):
    labels = config["labels"]
    labels["Sin detección"] = 0
    del labels["negative"]
        
    class_names = {v: k for k, v in labels.items()}
    return class_names



def plot_confusion_matrix(confusion_mat, class_names, figsize=(10, 8), cmap="Blues", save_path=None):
    """
    Visualiza la matriz de confusión de manera moderna con Seaborn.

    Args:
        confusion_mat (np.ndarray): Matriz de confusión.
        class_names (dict): Diccionario que mapea índices a nombres de clases.
        figsize (tuple, optional): Tamaño de la figura. Default (10, 8).
        cmap (str, optional): Colormap de Seaborn. Default "Blues".
        save_path (str, optional): Ruta para guardar la imagen. Si None, no se guarda.
    """
    # Convertir los índices a nombres de clases
    class_labels = [class_names[i] for i in range(len(class_names))]

    # Normalizar para mostrar porcentajes
    confusion_norm = confusion_mat.astype('float') / (confusion_mat.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=figsize)
    sns.heatmap(
        confusion_norm,
        annot=confusion_mat,  # Mostrar los conteos reales
        fmt="d",
        cmap=cmap,
        cbar=True,
        xticklabels=class_labels,
        yticklabels=class_labels,
    )

    plt.title("Matriz de Confusión", fontsize=16, weight="bold")
    plt.xlabel("Clases Predichas", fontsize=12)
    plt.ylabel("Clases Verdaderas", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def generate_roc_curve(y_true, y_scores, class_names, output_dir=None):
    y_true = np.array(y_true)
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr_list = []
    tpr_list = []
    roc_auc_list = []
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_scores[:, i])
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(auc(fpr, tpr))

    return fpr_list, tpr_list, roc_auc_list

def plot_roc_curve(y_true, y_scores, class_names, figsize=(10, 8), save_path=None):
    roc_curves = generate_roc_curve(y_true, y_scores, class_names)
    fpr_list, tpr_list, roc_auc_list = roc_curves

    for i in range(len(roc_auc_list)):
        plt.plot(fpr_list[i], tpr_list[i], label=f'{class_names[i]} (AUC = {roc_auc_list[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def save_confusion_matrix(confusion_mat, class_names, file_path):
    """
    Guarda la matriz de confusión con nombres de clases en un archivo de texto.
    
    Args:
        confusion_mat (np.ndarray): Matriz de confusión.
        class_names (dict): Diccionario que mapea índices a nombres de clases.
        file_path (str): Ruta del archivo donde se guardará la matriz.
    """
    # Convertir los índices a nombres de clases
    class_labels = [class_names[i] for i in range(len(class_names))]

    # Crear una representación de la matriz con nombres de clases
    with open(file_path, "w") as f:
        # Escribir encabezados
        f.write("\t" + "\t".join(class_labels) + "\n")
        for i, row in enumerate(confusion_mat):
            f.write(f"{class_labels[i]}\t" + "\t".join(map(str, row)) + "\n")

def tf_model_accuracy(
    config,
    folder,
    audio_processor,
    data_set="testing",
    accuracy_name="tf_model_accuracy.txt",
):
    class_names = get_classes(config)

    # Obtener datos de prueba
    test_x, test_y, _ = audio_processor.get_data(
        data_set,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy="truncate_start",
    )


    with tf.device("/cpu:0"):
        # Cargar el modelo TensorFlow guardado
        model = tf.saved_model.load(os.path.join(config["train_dir"], folder))
        infer = model.signatures["serving_default"]

        # Listas para almacenar etiquetas verdaderas y predicciones
        y_true, y_pred = [], []
        for i, sample in enumerate(test_x):
            tensor = tf.convert_to_tensor(sample[np.newaxis, ...], dtype=tf.float32)
            result = infer(tensor)
            probs = result["output"].numpy()[0]
            y_pred.append(np.argmax(probs))
            y_true.append(int(test_y[i]))

            if i % 1000 == 0 and i:
                logging.info(f"Procesando muestra {i} de {len(test_x)}")

    y_score = []
    for sample in test_x:
        spectrogram = sample.astype(np.float32)
        probs = model.predict_spectrogram(spectrogram)
        if probs.ndim == 2:
            probs = np.mean(probs, axis=0)
        y_score.append(probs)
    
    y_score = np.array(y_score)
    plot_roc_curve(
        y_true, y_score,
        class_names,
        save_path=os.path.join(config["train_dir"],
                               folder, "roc_curve.png"))

    # Calcular la matriz de confusión
    confusion_mat = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(confusion_mat, class_names)
    report = classification_report(
        y_true, y_pred,
        target_names=list(class_names.values()),
        zero_division=0,
        output_dict=True)

    accuracy = report["accuracy"]

    path = os.path.join(config["train_dir"], folder)
    with open(os.path.join(path, accuracy_name), "wt") as fd:
        fd.write(f"Accuracy: {accuracy:.4f}\n\n")
    
    with open(os.path.join(path, "classification_report.csv"), "wt") as fd:
        fd.write("label,precision,recall,f1-score\n")
        for label in class_names.values():
            if label in report:
                m = report[label]
                fd.write(f"{label},{m['precision']:.4f},{m['recall']:.4f},{m['f1-score']:.4f}\n")
    
    with open(os.path.join(path, "confusion_matrix.csv"), "wt") as fd:
        for i in range(confusion_mat.shape[0]):
            for j in range(confusion_mat.shape[1]):
                fd.write(f"{confusion_mat[i, j]} ")
            fd.write("\n")
        
    logging.info(f"Accuracy final en {data_set}: {accuracy:.4f}")

    return {"accuracy": accuracy, "report": report, "confusion_matrix": confusion_mat}


def tflite_streaming_model_roc(
    config,
    folder,
    audio_processor,
    data_set="testing",
    ambient_set="testing_ambient",
    tflite_model_name="stream_state_internal.tflite",
    accuracy_name="tflite_streaming_roc.txt",
    sliding_window_length=5,
):
    """Function to test a tflite model false accepts per hour and false rejection rates.

    Model can be streaming or nonstreaming. Nonstreaming models are strided by 1 spectrogram feature in the time dimension.

    Args:
        config (dict): dictionary containing microWakeWord training configuration
        folder (str): folder containing the TFLite model
        audio_processor (FeatureHandler): microWakeWord FeatureHandler object for retrieving spectrograms
        data_set (str, optional): Dataset for testing recall. Defaults to "testing".
        ambient_set (str, optional): Dataset for testing false accepts per hour. Defaults to "testing_ambient".
        tflite_model_name (str, optional): filename of the TFLite model. Defaults to "stream_state_internal.tflite".
        accuracy_name (str, optional): filename to save metrics at various cutoffs. Defaults to "tflite_streaming_roc.txt".
        sliding_window_length (int, optional): the length of the sliding window for computing average probabilities. Defaults to 1.

    Returns:
        float: The Area under the false accept per hour vs. false rejection curve.
    """
    stride = config["stride"]
    model = Model(
        os.path.join(config["train_dir"], folder, tflite_model_name), stride=stride
    )

    test_ambient_fingerprints, _, _ = audio_processor.get_data(
        ambient_set,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy="none",
    )

    logging.info("Testing the " + ambient_set + " set.")
    ambient_preds = []
    for spectrogram in test_ambient_fingerprints:
        probs = model.predict_spectrogram(spectrogram)  # (frames, num_classes)
        if probs.ndim == 2:  # suavizado temporal
            sw = sliding_window_view(probs, (sliding_window_length, probs.shape[1]), axis=(0, 1))
            moving_avg = sw.mean(axis=(2, 3))  # promedio sobre ventana
            probs = moving_avg
        ambient_preds.append(np.mean(probs, axis=0))  # promedio global

    test_fingerprints, test_ground_truth, _ = audio_processor.get_data(
        data_set,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy="none",
    )

    logging.info("Testing the " + data_set + " set.")
    y_true, y_pred = [], []
    for i in range(len(test_fingerprints)):
        probs = model.predict_spectrogram(test_fingerprints[i])
        if probs.ndim == 2:
            sw = sliding_window_view(probs, (sliding_window_length, probs.shape[1]), axis=(0, 1))
            moving_avg = sw.mean(axis=(2, 3))
            probs = moving_avg
        mean_probs = np.mean(probs, axis=0)
        y_pred.append(np.argmax(mean_probs))
        y_true.append(test_ground_truth[i])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = np.mean(y_true == y_pred)

    path = os.path.join(config["train_dir"], folder)
    with open(os.path.join(path, accuracy_name), "wt") as fd:
        acc_str = f"Accuracy: {accuracy:.4f}"
        logging.info(acc_str)
        fd.write(acc_str + "\n")

    return {"accuracy": accuracy}

def tflite_model_accuracy(
    config,
    folder,
    audio_processor,
    data_set="testing",
    tflite_model_name="stream_state_internal.tflite",
    accuracy_name="tflite_model_accuracy.txt",
):
    """
    Evalúa la precisión de un modelo TFLite siguiendo el flujo de tf_model_accuracy.

    Args:
        config (dict): Configuración de entrenamiento.
        folder (str): Carpeta donde está el modelo.
        audio_processor (FeatureHandler): Manejador de características.
        data_set (str, optional): Conjunto de datos a evaluar. Default "testing".
        tflite_model_name (str, optional): Nombre del archivo TFLite. Default "stream_state_internal.tflite".
        accuracy_name (str, optional): Nombre del archivo de reporte. Default "tflite_model_accuracy.txt".

    Returns:
        dict: Contiene accuracy, reporte de clasificación y matriz de confusión.
    """
    class_names = get_classes(config)

    # Cargar modelo TFLite
    model = Model(os.path.join(config["train_dir"], folder, tflite_model_name))

    # Estrategia de truncamiento
    truncation_strategy = "truncate_start"
    if data_set.endswith("ambient"):
        truncation_strategy = "none"

    # Obtener datos de prueba
    test_x, test_y, _ = audio_processor.get_data(
        data_set,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy=truncation_strategy,
    )

    y_true, y_pred = [], []

    for i, sample in enumerate(test_x):
        spectrogram = sample.astype(np.float32)
        probs = model.predict_spectrogram(spectrogram)
        if probs.ndim == 2:
            # Promediar temporalmente si es necesario
            probs = np.mean(probs, axis=0)
        pred_class = int(np.argmax(probs))
        y_pred.append(pred_class)
        y_true.append(int(test_y[i]))

        if i % 1000 == 0 and i:
            logging.info(f"Procesando muestra {i} de {len(test_x)}")

    # Calcular métricas
    confusion_mat = confusion_matrix(y_true, y_pred)   
    plot_confusion_matrix(confusion_mat, class_names)
    
    report = classification_report(
        y_true, y_pred,
        target_names=list(class_names.values()),
        output_dict=True,
        zero_division=0,
    )
    accuracy = report["accuracy"]

    # Guardar reporte y matriz
    path = os.path.join(config["train_dir"], folder)
    with open(os.path.join(path, accuracy_name), "wt") as fd:
        fd.write(f"Accuracy: {accuracy:.4f}\n\n")
        fd.write("Reporte de clasificación:\n")
        for label in class_names.values():
            if label in report:
                m = report[label]
                fd.write(
                    f"{label}: precision={m['precision']:.4f}, recall={m['recall']:.4f}, f1={m['f1-score']:.4f}\n"
                )

        fd.write("\nMatriz de confusión:\n")
        np.savetxt(fd, confusion_mat, fmt="%d")

    # Visualizar matriz de confusión con la función moderna
    plot_confusion_matrix(
        confusion_mat,
        class_names,
        figsize=(10, 8),
        cmap="Blues",
        save_path=os.path.join(path, "confusion_matrix.png"),
    )

    logging.info(f"Accuracy final en {data_set}: {accuracy:.4f}")

    return {"accuracy": accuracy, "report": report, "confusion_matrix": confusion_mat}
