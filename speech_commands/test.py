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

from absl import logging
from typing import List
from speech_commands.inference import Model
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import confusion_matrix


def get_classes(config):
    labels = os.listdir(config["commands_dir"])
    class_names = {idx + 1: label for idx, label in enumerate(labels)}
    class_names[0] = "Inaplicable"
    return class_names



def plot_confusion_matrix(confusion_mat, class_names):
    """
    Visualiza la matriz de confusión con nombres de clases.
    
    Args:
        confusion_mat (np.ndarray): Matriz de confusión.
        class_names (dict): Diccionario que mapea índices a nombres de clases.
    """
    # Convertir los índices a nombres de clases
    class_labels = [class_names[i] for i in range(len(class_names))]

    # Visualizar la matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_labels,  # Etiquetas del eje x
        yticklabels=class_labels,  # Etiquetas del eje y
    )
    plt.xlabel("Clases Predichas")
    plt.ylabel("Clases Verdaderas")
    plt.title("Matriz de Confusión")
    plt.xticks(rotation=45)  # Rotar etiquetas si son largas
    plt.yticks(rotation=0)
    plt.savefig("confusion_matrix.png")
    plt.show()


def compute_metrics_from_confusion(confusion_mat):
    """
    Calcula métricas globales (precisión, recall, etc.) desde una matriz de confusión.
    """
    true_positives = np.diag(confusion_mat)
    false_positives = np.sum(confusion_mat, axis=0) - true_positives
    false_negatives = np.sum(confusion_mat, axis=1) - true_positives

    precision = np.mean(true_positives / (true_positives + false_positives + 1e-9))
    recall = np.mean(true_positives / (true_positives + false_negatives + 1e-9))
    accuracy = np.sum(true_positives) / np.sum(confusion_mat)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }

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


def compute_metrics_multiclass(confusion_mat):
    """
    Calcula métricas globales y por clase a partir de una matriz de confusión multiclase.
    
    Args:
        confusion_mat (np.ndarray): Matriz de confusión de tamaño (num_classes x num_classes).
    
    Returns:
        dict: Diccionario con métricas globales y por clase.
    """
    num_classes = confusion_mat.shape[0]
    true_positives = np.diag(confusion_mat)  # Diagonal contiene los verdaderos positivos
    false_positives = np.sum(confusion_mat, axis=0) - true_positives  # Suma por columnas
    false_negatives = np.sum(confusion_mat, axis=1) - true_positives  # Suma por filas
    true_negatives = np.sum(confusion_mat) - (true_positives + false_positives + false_negatives)

    # Métricas por clase
    precision_per_class = true_positives / (true_positives + false_positives + 1e-9)
    recall_per_class = true_positives / (true_positives + false_negatives + 1e-9)
    f1_score_per_class = (
        2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + 1e-9)
    )

    # Métricas globales
    accuracy = np.sum(true_positives) / np.sum(confusion_mat)
    precision_macro = np.mean(precision_per_class)
    recall_macro = np.mean(recall_per_class)
    f1_score_macro = np.mean(f1_score_per_class)

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_score_macro": f1_score_macro,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_score_per_class": f1_score_per_class,
    }

def compute_metrics(true_positives, true_negatives, false_positives, false_negatives):
    """Utility function to compute various metrics.

    Arguments:
        true_positives: Count of samples correctly predicted as positive
        true_negatives: Count of samples correctly predicted as negative
        false_positives: Count of samples incorrectly predicted as positive
        false_negatives: Count of samples incorrectly predicted as negative

    Returns:
        metric dictionary with keys for `accuracy`, `recall`, `precision`, `false_positive_rate`, `false_negative_rate`, and `count`
    """

    accuracy = float("nan")
    false_positive_rate = float("nan")
    false_negative_rate = float("nan")
    recall = float("nan")
    precision = float("nan")

    count = true_positives + true_negatives + false_positives + false_negatives

    if true_positives + true_negatives + false_positives + false_negatives > 0:
        accuracy = (true_positives + true_negatives) / count

    if false_positives + true_negatives > 0:
        false_positive_rate = false_positives / (false_positives + true_negatives)

    if true_positives + false_negatives > 0:
        false_negative_rate = false_negatives / (true_positives + false_negatives)
        recall = true_positives / (true_positives + false_negatives)

    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives)

    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "count": count,
    }


def metrics_to_string(metrics):
    """Utility function to return a string that describes various metrics.

    Arguments:
        metrics: metric dictionary with keys for `accuracy`, `recall`, `precision`, `false_positive_rate`, `false_negative_rate`, and `count`

    Returns:
        string describing the given metrics
    """

    return "accuracy = {accuracy:.4%}; recall = {recall:.4%}; precision = {precision:.4%}; fpr = {fpr:.4%}; fnr = {fnr:.4%}; (N={count})".format(
        accuracy=metrics["accuracy"],
        recall=metrics["recall"],
        precision=metrics["precision"],
        fpr=metrics["false_positive_rate"],
        fnr=metrics["false_negative_rate"],
        count=metrics["count"],
    )


def compute_false_accepts_per_hour(
    streaming_probabilities_list: List[np.ndarray],
    cutoffs: np.array,
    ignore_slices_after_accept: int = 75,
    stride: int = 1,
    step_s: float = 0.02,
):
    """Computes the false accept per hour rates at various cutoffs given a list of streaming probabilities.

    Args:
        streaming_probabilities_list (List[numpy.ndarray]): A list containing streaming probabilities from negative audio clips
        cutoffs (numpy.array): An array of cutoffs/thresholds to test the false accpet rate at.
        ignore_slices_after_accept (int, optional): The number of probabililities slices to ignore after a false accept. Defaults to 75.
        stride (int, optional): The stride of the input layer. Defaults to 1.
        step_s (float, optional): The duration between each probabilitiy in seconds. Defaults to 0.02.

    Returns:
        numpy.ndarray: The false accepts per hour corresponding to thresholds in `cutoffs`.
    """
    cutoffs_count = cutoffs.shape[0]

    false_accepts_at_cutoffs = np.zeros(cutoffs_count)
    probabilities_duration_h = 0

    for track_probabilities in streaming_probabilities_list:
        probabilities_duration_h += len(track_probabilities) * stride * step_s / 3600.0

        cooldown_at_cutoffs = np.ones(cutoffs_count) * ignore_slices_after_accept

        for wakeword_probability in track_probabilities:
            # Decrease the cooldown cutoff by 1 with a minimum value of 0
            cooldown_at_cutoffs = np.maximum(
                cooldown_at_cutoffs - 1, np.zeros(cutoffs_count)
            )
            detection_boolean = (
                wakeword_probability > cutoffs
            )  # a list of detection states at each cutoff

            for index in range(cutoffs_count):
                if cooldown_at_cutoffs[index] == 0 and detection_boolean[index]:
                    false_accepts_at_cutoffs[index] += 1
                    cooldown_at_cutoffs[index] = ignore_slices_after_accept

    return false_accepts_at_cutoffs / probabilities_duration_h


def generate_roc_curve(
    false_accepts_per_hour: np.ndarray,
    false_rejections: np.ndarray,
    # positive_samples_probabilities: np.ndarray,
    cutoffs: np.ndarray,
    max_faph: float = 2.0,
):
    """Generates the coordinates for an ROC curve plotting false accepts per hour vs false rejections. Computes the false rejection rate at the specifiied cutoffs.

    Args:
        false_accepts_per_hour (numpy.ndarray): False accepts per hour rates for each threshold in `cutoffs`.
        false_rejections (numpy.ndarray): False rejection rates for each threshold in `cutoffs`.
        cutoffs (numpy.ndarray): Thresholds used for `false_ccepts_per_hour`
        max_faph (float, optional): The maximum false accept per hour rate to include in curve's coordinates. Defaults to 2.0.

    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray): (false accept per hour coordinates, false rejection rate coordinates, cutoffs for each coordinate)
    """

    if false_accepts_per_hour[0] > max_faph:
        # Use linear interpolation to estimate false negative rate at max_faph

        # Increase the index until we find a faph less than max_faph
        index_of_first_viable = 1
        while false_accepts_per_hour[index_of_first_viable] > max_faph:
            index_of_first_viable += 1

        x0 = false_accepts_per_hour[index_of_first_viable - 1]
        y0 = false_rejections[index_of_first_viable - 1]
        x1 = false_accepts_per_hour[index_of_first_viable]
        y1 = false_rejections[index_of_first_viable - 1]

        fnr_at_max_faph = (y0 * (x1 - 2.0) + y1 * (2.0 - x0)) / (x1 - x0)
        cutoff_at_max_faph = (
            cutoffs[index_of_first_viable] + cutoffs[index_of_first_viable - 1]
        ) / 2.0
    else:
        # Smallest faph is less than max_faph, so assume the false negative rate is constant
        index_of_first_viable = 0
        fnr_at_max_faph = false_rejections[index_of_first_viable]
        cutoff_at_max_faph = cutoffs[index_of_first_viable]

    horizontal_coordinates = [max_faph]
    vertical_coordinates = [fnr_at_max_faph]
    cutoffs_at_coordinate = [cutoff_at_max_faph]

    for index in range(index_of_first_viable, len(false_rejections)):
        if false_accepts_per_hour[index] != horizontal_coordinates[-1]:
            # Only add a point if it is a new faph
            # This ensures if a faph rate is repeated, we use the small false negative rate
            horizontal_coordinates.append(false_accepts_per_hour[index])
            vertical_coordinates.append(false_rejections[index])
            cutoffs_at_coordinate.append(cutoffs[index])

    if horizontal_coordinates[-1] > 0:
        # If there isn't a cutoff with 0 faph, then add a coordinate at (0,1)
        horizontal_coordinates.append(0.0)
        vertical_coordinates.append(1.0)
        cutoffs_at_coordinate.append(0.0)

    # The points on the curve are listed in descending order, flip them before returning
    horizontal_coordinates = np.flip(horizontal_coordinates)
    vertical_coordinates = np.flip(vertical_coordinates)
    cutoffs_at_coordinate = np.flip(cutoffs_at_coordinate)
    return horizontal_coordinates, vertical_coordinates, cutoffs_at_coordinate


import librosa
import numpy as np

def mel_to_audio(mel_spectrogram, sr=16000, n_fft=512, step_ms=10, n_mels=40):
    """
    Convierte un espectrograma Mel en una señal de audio.
    
    Args:
        mel_spectrogram (np.ndarray): Espectrograma Mel (shape: [tiempo, n_mels] o [n_mels, tiempo]).
        sr (int): Tasa de muestreo del audio original.
        n_fft (int): Tamaño de la ventana FFT.
        step_ms (int): Desplazamiento entre ventanas consecutivas (en milisegundos).
        n_mels (int): Número de bandas Mel.
    
    Returns:
        np.ndarray: Señal de audio reconstruida.
    """
    # Calcular hop_length a partir de step_ms
    # hop_length = int(step_ms * sr / 1000)
    hop_length = mel_spectrogram.shape[0]

    if mel_spectrogram.shape[1] == n_mels:  # Si la segunda dimensión es n_mels
        mel_spectrogram = mel_spectrogram.T  # Transponer el tensor
    print("Forma del espectrograma Mel ajustado:", mel_spectrogram.shape)

    # Crear la matriz de filtro Mel inversa
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=125,  # Límite inferior de frecuencia
        fmax=7500,  # Límite superior de frecuencia
    )
    mel_inverse = np.linalg.pinv(mel_basis)

    # Convertir características Mel a espectrograma de magnitud
    magnitude_spectrogram = np.dot(mel_inverse, mel_spectrogram)

    # Aplicar Griffin-Lim para reconstruir la fase
    audio_signal = librosa.griffinlim(
        magnitude_spectrogram,
        n_iter=10,  # Número de iteraciones para Griffin-Lim
        hop_length=hop_length,
        win_length=n_fft
    )

    return audio_signal


def tf_model_accuracy(
    config,
    folder,
    audio_processor,
    data_set="testing",
    accuracy_name="tf_model_accuracy.txt",
):
    # Obtener datos de prueba
    test_fingerprints, test_ground_truth, _ = audio_processor.get_data(
        data_set,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy="truncate_start",
    )

    class_names = get_classes(config)

    with tf.device("/cpu:0"):
        # Cargar el modelo TensorFlow guardado
        model = tf.saved_model.load(os.path.join(config["train_dir"], folder))
        inference_batch_size = 1

        infer = model.signatures["serving_default"]

        # Listas para almacenar etiquetas verdaderas y predicciones
        y_true = []
        y_pred = []


        for i in range(0, len(test_fingerprints)):
            spectrogram_features = test_fingerprints[i : i + inference_batch_size]  # Tomar una muestra
            spectrogram_tensor = tf.convert_to_tensor(spectrogram_features, dtype=tf.float32)
            sample_ground_truth = int(test_ground_truth[i])


            result = infer(spectrogram_tensor)
            probabilities = result['output_0'].numpy()[0]  # Probabilidades para cada clase
            predicted_class = np.argmax(probabilities)  # Clase con mayor probabilidad
            
            # Almacenar etiquetas verdaderas y predicciones
            y_true.append(sample_ground_truth)
            y_pred.append(predicted_class)

            # Registro intermedio (opcional)
            if i % 1000 == 0 and i:
                logging.info(f"Procesando muestra {i} de {len(test_fingerprints)}")
    

    # Calcular la matriz de confusión
    confusion_mat = confusion_matrix(y_true, y_pred)

    # Visualizar la matriz de confusión
    plot_confusion_matrix(confusion_mat, class_names)


    # Guardar resultados
    metrics_string = compute_metrics_from_confusion(confusion_mat)
    logging.info("Final TensorFlow model on the " + data_set + " set: " + metrics_string)
    path = os.path.join(config["train_dir"], folder)
    with open(os.path.join(path, accuracy_name), "wt") as fd:
        fd.write(metrics_string + "\n")
        fd.write("Matriz de confusión:\n")
        np.savetxt(fd, confusion_mat, fmt="%d")

    return compute_metrics_from_confusion(confusion_mat)


def tflite_streaming_model_roc(
    config,
    folder,
    audio_processor,
    data_set="testing",
    ambient_set="testing_ambient",
    tflite_model_name="stream_state_internal.tflite",
    accuracy_name="tflite_streaming_roc.txt",
    sliding_window_length=5,
    ignore_slices_after_accept=25,
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
    ambient_streaming_probabilities = []
    for spectrogram_track in test_ambient_fingerprints:
        streaming_probabilities = model.predict_spectrogram(spectrogram_track)
        sliding_window_probabilities = sliding_window_view(
            streaming_probabilities, sliding_window_length
        )
        moving_average = sliding_window_probabilities.mean(axis=-1)
        ambient_streaming_probabilities.append(moving_average)

    cutoffs = np.arange(0, 1.01, 0.01)
    # ignore_slices_after_accept = 25

    faph = compute_false_accepts_per_hour(
        ambient_streaming_probabilities,
        cutoffs,
        ignore_slices_after_accept,
        stride=config["stride"],
        step_s=config["window_step_ms"] / 1000,
    )

    test_fingerprints, test_ground_truth, _ = audio_processor.get_data(
        data_set,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy="none",
    )

    logging.info("Testing the " + data_set + " set.")

    positive_sample_streaming_probabilities = []
    for i in range(len(test_fingerprints)):
        if test_ground_truth[i]:
            # Only test positive samples
            streaming_probabilities = model.predict_spectrogram(test_fingerprints[i])
            sliding_window_probabilities = sliding_window_view(
                streaming_probabilities[ignore_slices_after_accept:],
                sliding_window_length,
            )
            moving_average = sliding_window_probabilities.mean(axis=-1)
            positive_sample_streaming_probabilities.append(np.max(moving_average))

    # Compute the false negative rates at each cutoff
    false_negative_rate_at_cutoffs = []
    for cutoff in cutoffs:
        true_accepts = sum(i > cutoff for i in positive_sample_streaming_probabilities)
        false_negative_rate_at_cutoffs.append(
            1 - true_accepts / len(positive_sample_streaming_probabilities)
        )

    x_coordinates, y_coordinates, cutoffs_at_points = generate_roc_curve(
        false_accepts_per_hour=faph,
        false_rejections=false_negative_rate_at_cutoffs,
        cutoffs=cutoffs,
    )

    path = os.path.join(config["train_dir"], folder)
    with open(os.path.join(path, accuracy_name), "wt") as fd:
        auc = np.trapz(y_coordinates, x_coordinates)
        auc_string = "AUC {:.5f}".format(auc)
        logging.info(auc_string)
        fd.write(auc_string + "\n")

        for i in range(0, x_coordinates.shape[0]):
            cutoff_string = "Cutoff {:.2f}: frr={:.4f}; faph={:.3f}".format(
                cutoffs_at_points[i], y_coordinates[i], x_coordinates[i]
            )
            logging.info(cutoff_string)
            fd.write(cutoff_string + "\n")

    return auc


def tflite_model_accuracy(
    config,
    folder,
    audio_processor,
    data_set="testing",
    tflite_model_name="stream_state_internal.tflite",
    accuracy_name="tflite_model_accuracy.txt",
):
    # Cargar el modelo TFLite
    model = Model(os.path.join(config["train_dir"], folder, tflite_model_name))

    # Determinar la estrategia de truncamiento
    truncation_strategy = "truncate_start"
    if data_set.endswith("ambient"):
        truncation_strategy = "none"

    # Obtener datos de prueba
    test_fingerprints, test_ground_truth, _ = audio_processor.get_data(
        data_set,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy=truncation_strategy,
    )

    logging.info(f"Testing TFLite model on the {data_set} set")

    # Listas para almacenar etiquetas verdaderas y predicciones
    y_true = []
    y_pred = []
    class_names = get_classes(config)

    for i in range(len(test_fingerprints)):
        spectrogram = test_fingerprints[i].astype(np.float32)
        probabilities = model.predict_spectrogram(spectrogram)
        print("Holaaaaaaaaaaaa")
        predicted_class = np.argmax(probabilities[-1])  # Clase con mayor probabilidad
        print(f"Clase predicha para la muestra {i}: {predicted_class}")

        # Almacenar etiquetas verdaderas y predicciones
        true_class = int(test_ground_truth[i])  # Convertir a entero si es necesario
        y_true.append(true_class)
        y_pred.append(predicted_class)

        # Registro intermedio (opcional)
        if i % 1000 == 0 and i:
            logging.info(f"Procesando muestra {i} de {len(test_fingerprints)}")

    # Calcular la matriz de confusión
    confusion_mat = confusion_matrix(y_true, y_pred)
    print("Matriz de confusión:\n", confusion_mat)

    class_names = get_classes(config)
    plot_confusion_matrix(confusion_mat, class_names)

    # Calcular métricas globales y por clase
    metrics = compute_metrics_multiclass(confusion_mat)

    # Guardar resultados
    metrics_string = (
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"Precisión macro: {metrics['precision_macro']:.4f}\n"
        f"Recall macro: {metrics['recall_macro']:.4f}\n"
        f"F1-score macro: {metrics['f1_score_macro']:.4f}"
    )
    logging.info(f"Final TFLite model on the {data_set} set:\n{metrics_string}")
    path = os.path.join(config["train_dir"], folder)
    with open(os.path.join(path, accuracy_name), "wt") as fd:
        fd.write(metrics_string + "\n")
        fd.write("Matriz de confusión:\n")
        np.savetxt(fd, confusion_mat, fmt="%d")

    return metrics
