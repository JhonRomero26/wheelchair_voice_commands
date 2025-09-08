```data.py
# coding=utf-8
# Copyright 2024 Kevin Ahrendt.
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

"""Functions and classes for loading/augmenting spectrograms"""

import os
import random

import numpy as np

from absl import logging
from pathlib import Path
from mmap_ninja.ragged import RaggedMmap

from speech_commands.audio.clips import Clips
from speech_commands.audio.augmentation import Augmentation
from speech_commands.audio.spectrograms import SpectrogramGeneration


def spec_augment(
    spectrogram: np.ndarray,
    time_mask_max_size: int = 0,
    time_mask_count: int = 0,
    freq_mask_max_size: int = 0,
    freq_mask_count: int = 0,
):
    """Applies SpecAugment to the input spectrogram.
    Based on SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition by D. Park, W. Chan, Y. Zhang, C. Chiu, B. Zoph, E Cubuk, Q Le
    https://arxiv.org/pdf/1904.08779.pdf
    Implementation based on https://github.com/pyyush/SpecAugment/tree/master

    Args:
        spectrogram (numpy.ndarray): The input spectrogram.
        time_mask_max_size (int): The maximum size of time feature masks. Defaults to 0.
        time_mask_count (int): The total number of separate time masks. Defaults to 0.
        freq_mask_max_size (int): The maximum size of frequency feature masks. Defaults to 0.
        time_mask_count (int): The total number of separate feature masks. Defaults to 0.

    Returns:
        numpy.ndarray: The masked spectrogram.
    """

    time_frames = spectrogram.shape[0]
    freq_bins = spectrogram.shape[1]

    # Spectrograms yielded from a generator are read only
    augmented_spectrogram = np.copy(spectrogram)

    for i in range(time_mask_count):
        t = int(np.random.uniform(0, time_mask_max_size))
        t0 = random.randint(0, time_frames - t)
        augmented_spectrogram[t0 : t0 + t, :] = 0

    for i in range(freq_mask_count):
        f = int(np.random.uniform(0, freq_mask_max_size))
        f0 = random.randint(0, freq_bins - f)
        augmented_spectrogram[:, f0 : f0 + f] = 0

    return augmented_spectrogram


def fixed_length_spectrogram(
    spectrogram: np.ndarray,
    features_length: int,
    truncation_strategy: str = "random",
    right_cutoff: int = 0,
):
    """Returns a spectrogram with specified length. Pads with zeros at the start if too short. Removes feature windows following ``truncation_strategy`` if too long.

    Args:
        spectrogram (numpy.ndarray): The spectrogram to truncate or pad.
        features_length (int): The desired spectrogram length.
        truncation_strategy (str): How to truncate if ``spectrogram`` is longer than ``features_length`` One of:
            random: choose a random portion of the entire spectrogram - useful for long negative samples
            truncate_start: remove the start of the spectrogram
            truncate_end: remove the end of the spectrogram
            none: returns the entire spectrogram regardless of features_length


    Returns:
        numpy.ndarry: The fixed length spectrogram due to padding or truncation.
    """

    data_length = spectrogram.shape[0]
    features_offset = 0
    if data_length > features_length:
        if truncation_strategy == "random":
            features_offset = np.random.randint(0, data_length - features_length)
        elif truncation_strategy == "none":
            # return the entire spectrogram
            features_length = data_length
        elif truncation_strategy == "truncate_start":
            features_offset = data_length - features_length
        elif truncation_strategy == "truncate_end":
            features_offset = 0
        elif truncation_strategy == "fixed_right_cutoff":
            features_offset = data_length - features_length - right_cutoff
    else:
        pad_slices = features_length - data_length

        spectrogram = np.pad(
            spectrogram, ((pad_slices, 0), (0, 0)), constant_values=(0, 0)
        )
        features_offset = 0

    return spectrogram[features_offset : (features_offset + features_length)]


class MmapFeatureGenerator(object):
    """A class that handles loading spectrograms from Ragged MMaps for training or testing.

    Args:
        path (str): Input directory to the Ragged MMaps. The Ragged MMap folders should be included in the following file structure:
            training/ (spectrograms to use for training the model)
            validation/ (spectrograms used to validate the model while training)
            testing/ (spectrograms used to test the model after training)
            validation_ambient/ (spectrograms of long duration background audio clips that are split and validated while training)
            testing_ambient/ (spectrograms of long duration background audio clips to test the model after training)
        label (bool): The class each spectrogram represents; i.e., wakeword or not.
        sampling_weight (float): The sampling weight for how frequently a spectrogram from this dataset is chosen.
        penalty_weight (float): The penalizing weight for incorrect predictions for each spectrogram.
        truncation_strategy (str): How to truncate if ``spectrogram`` is too long.
        stride (int): The stride in the model's first layer.
        step (float): The window step duration (in seconds).
        fixed_right_cutoffs (list[int]): List of spectogram slices to cutoff on the right if the truncation strategy is "fixed_right_cutoff". In training mode, its randomly chosen from the list. Otherwise, it yields spectrograms with all cutoffs in the list.
    """

    def __init__(
        self,
        path: str,
        label: int,
        sampling_weight: float,
        penalty_weight: float,
        truncation_strategy: str,
        stride: int,
        step: float,
        fixed_right_cutoffs: list[int] = [0],
    ):
        self.label = int(label)
        self.sampling_weight = sampling_weight
        self.penalty_weight = penalty_weight
        self.truncation_strategy = truncation_strategy
        self.fixed_right_cutoffs = fixed_right_cutoffs

        self.stride = stride
        self.step = step

        self.stats = {}
        self.feature_sets = {}

        self.feature_sets["testing"] = []
        self.feature_sets["training"] = []
        self.feature_sets["validation"] = []
        self.feature_sets["validation_ambient"] = []
        self.feature_sets["testing_ambient"] = []

        self.loaded_features = []

        dirs = [
            "testing",
            "training",
            "validation",
            "testing_ambient",
            "validation_ambient",
        ]

        for set_index in dirs:
            duration = 0.0
            count = 0

            search_path_directory = os.path.join(path, set_index)
            search_path = [
                str(i)
                for i in Path(os.path.abspath(search_path_directory)).glob("**/*_mmap/")
            ]

            for mmap_path in search_path:
                imported_features = RaggedMmap(mmap_path)

                self.loaded_features.append(imported_features)
                feature_index = len(self.loaded_features) - 1

                for i in range(0, len(imported_features)):
                    self.feature_sets[set_index].append(
                        {
                            "loaded_feature_index": feature_index,
                            "subindex": i,
                        }
                    )

                    duration += step * imported_features[i].shape[0]
                    count += 1

            random.shuffle(self.feature_sets[set_index])

            self.stats[set_index] = {
                "spectrogram_count": count,
                "total_duration": duration,
            }

    def get_mode_duration(self, mode: str):
        """Retrieves the total duration of the spectrograms in the mode set.

        Args:
            mode (str): Specifies the set. One of "training", "validation", "testing", "validation_ambient", "testing_ambient".

        Returns:
            float: The duration in hours.
        """
        return self.stats[mode]["total_duration"]

    def get_mode_size(self, mode):
        """Retrieves the total count of the spectrograms in the mode set.

        Args:
            mode (str): Specifies the set. One of "training", "validation", "testing", "validation_ambient", "testing_ambient".

        Returns:
            int: The spectrogram count.
        """
        return self.stats[mode]["spectrogram_count"]

    def get_random_spectrogram(
        self, mode: str, features_length: int, truncation_strategy: str
    ):
        """Retrieves a random spectrogram from the specified mode with specified length after truncation.

        Args:
            mode (str): Specifies the set. One of "training", "validation", "testing", "validation_ambient", "testing_ambient".
            features_length (int): The length of the spectrogram in feature windows.
            truncation_strategy (str): How to truncate if ``spectrogram`` is too long.

        Returns:
            numpy.ndarray: A random spectrogram of specified length after truncation.
        """
        right_cutoff = 0
        if truncation_strategy == "default":
            truncation_strategy = self.truncation_strategy

        if truncation_strategy == "fixed_right_cutoff":
            right_cutoff = random.choice(self.fixed_right_cutoffs)

        feature = random.choice(self.feature_sets[mode])
        spectrogram = self.loaded_features[feature["loaded_feature_index"]][
            feature["subindex"]
        ]

        spectrogram = fixed_length_spectrogram(
            spectrogram,
            features_length,
            truncation_strategy,
            right_cutoff,
        )

        # Spectrograms with type np.uint16 haven't been scaled
        if np.issubdtype(spectrogram.dtype, np.uint16):
            spectrogram = spectrogram.astype(np.float32) * 0.0390625

        return spectrogram

    def get_feature_generator(
        self,
        mode,
        features_length,
        truncation_strategy="default",
    ):
        """A Python generator that yields spectrograms from the specified mode of specified length after truncation.

        Args:
            mode (str): Specifies the set. One of "training", "validation", "testing", "validation_ambient", "testing_ambient".
            features_length (int): The length of the spectrogram in feature windows.
            truncation_strategy (str): How to truncate if ``spectrogram`` is too long.

        Yields:
            numpy.ndarray: A random spectrogram of specified length after truncation.
        """
        if truncation_strategy == "default":
            truncation_strategy = self.truncation_strategy

        for feature in self.feature_sets[mode]:
            spectrogram = self.loaded_features[feature["loaded_feature_index"]][
                feature["subindex"]
            ]

            # Spectrograms with type np.uint16 haven't been scaled
            if np.issubdtype(spectrogram.dtype, np.uint16):
                spectrogram = spectrogram.astype(np.float32) * 0.0390625

            if truncation_strategy == "split":
                for feature_start_index in range(
                    0,
                    spectrogram.shape[0] - features_length,
                    int(1000 * self.step * self.stride),
                ):  # 10*2 features corresponds to 200 ms
                    split_spectrogram = spectrogram[
                        feature_start_index : feature_start_index + features_length
                    ]

                    yield split_spectrogram
            else:
                for cutoff in self.fixed_right_cutoffs:
                    fixed_spectrogram = fixed_length_spectrogram(
                        spectrogram,
                        features_length,
                        truncation_strategy,
                        cutoff,
                    )

                    yield fixed_spectrogram


class ClipsHandlerWrapperGenerator(object):
    """A class that handles loading spectrograms from audio files on the disk to use while training. This generates spectrograms with random augmentations applied during the training process.

    Args:
        spectrogram_generation (SpectrogramGeneration): Object that handles generating spectrograms from audio files.
        label (bool): The class each spectrogram represents; i.e., wakeword or not.
        sampling_weight (float): The sampling weight for how frequently a spectrogram from this dataset is chosen.
        penalty_weight (float): The penalizing weight for incorrect predictions for each spectrogram.
        truncation_strategy (str): How to truncate if ``spectrogram`` is too long.
    """

    def __init__(
        self,
        spectrogram_generation: SpectrogramGeneration,
        label: int,
        sampling_weight: float,
        penalty_weight: float,
        truncation_strategy: str,
    ):
        self.spectrogram_generation = spectrogram_generation
        self.label = label
        self.sampling_weight = sampling_weight
        self.penalty_weight = penalty_weight
        self.truncation_strategy = truncation_strategy

        self.augmented_generator = self.spectrogram_generation.spectrogram_generator(
            random=True
        )

    def get_mode_duration(self, mode):
        """Function to maintain compatability with the MmapFeatureGenerator class."""
        return 0.0

    def get_mode_size(self, mode):
        """Function to maintain compatability with the MmapFeatureGenerator class. This class is intended only for retrieving spectrograms for training."""
        if mode == "training":
            return len(self.spectrogram_generation.clips.clips)
        else:
            return 0

    def get_random_spectrogram(self, mode, features_length, truncation_strategy):
        """Retrieves a random spectrogram from the specified mode with specified length after truncation.

        Args:
            mode (str): Specifies the set, but is ignored for this class. It is assumed the spectrograms will be for training.
            features_length (int): The length of the spectrogram in feature windows.
            truncation_strategy (str): How to truncate if ``spectrogram`` is too long.

        Returns:
            numpy.ndarray: A random spectrogram of specified length after truncation.
        """

        if truncation_strategy == "default":
            truncation_strategy = self.truncation_strategy

        spectrogram = next(self.augmented_generator)

        spectrogram = fixed_length_spectrogram(
            spectrogram,
            features_length,
            truncation_strategy,
            right_cutoff=0,
        )

        # Spectrograms with type np.uint16 haven't been scaled
        if np.issubdtype(spectrogram.dtype, np.uint16):
            spectrogram = spectrogram.astype(np.float32) * 0.0390625

        return spectrogram

    def get_feature_generator(
        self,
        mode,
        features_length,
        truncation_strategy="default",
    ):
        """Function to maintain compatability with the MmapFeatureGenerator class."""
        for x in []:
            yield x


class FeatureHandler(object):
    """Class that handles loading spectrogram features and providing them to the training and testing functions.

    Args:
      config: dictionary containing microWakeWord training configuration
    """

    def __init__(
        self,
        config: dict,
    ):
        self.feature_providers = []

        logging.info("Loading and analyzing data sets.")

        for feature_set in config["features"]:
            if feature_set["type"] == "mmap":
                self.feature_providers.append(
                    MmapFeatureGenerator(
                        feature_set["features_dir"],
                        feature_set["label"],
                        feature_set["sampling_weight"],
                        feature_set["penalty_weight"],
                        feature_set["truncation_strategy"],
                        stride=config["stride"],
                        step=config["window_step_ms"] / 1000.0,
                        fixed_right_cutoffs=feature_set.get("fixed_right_cutoffs", [0]),
                    )
                )
            elif feature_set["type"] == "clips":
                clips_handler = Clips(**feature_set["clips_settings"])
                augmentation_applier = Augmentation(
                    **feature_set["augmentation_settings"]
                )
                spectrogram_generator = SpectrogramGeneration(
                    clips_handler,
                    augmentation_applier,
                    **feature_set["spectrogram_generation_settings"],
                )
                self.feature_providers.append(
                    ClipsHandlerWrapperGenerator(
                        spectrogram_generator,
                        feature_set["label"],
                        feature_set["sampling_weight"],
                        feature_set["penalty_weight"],
                        feature_set["truncation_strategy"],
                    )
                )
            set_modes = [
                "training",
                "validation",
                "testing",
                "validation_ambient",
                "testing_ambient",
            ]
            total_spectrograms = 0
            for set in set_modes:
                total_spectrograms += self.feature_providers[-1].get_mode_size(set)

            if total_spectrograms == 0:
                logging.warning("No spectrograms found in a configured feature set:")
                logging.warning(feature_set)

    def get_mode_duration(self, mode: str):
        """Returns the durations of all spectrogram features in the given mode.

        Args:
            mode (str): which training set to compute duration over. One of `training`, `testing`, `testing_ambient`, `validation`, or `validation_ambient`

        Returns:
            duration, in seconds, of all spectrograms in this mode
        """

        sample_duration = 0
        for provider in self.feature_providers:
            sample_duration += provider.get_mode_duration(mode)
        return sample_duration

    def get_mode_size(self, mode: str):
        """Returns the count of all spectrogram features in the given mode.

        Args:
            mode (str): which training set to count the spectrograms. One of `training`, `testing`, `testing_ambient`, `validation`, or `validation_ambient`

        Returns:
            count of spectrograms in given mode
        """
        sample_count = 0
        for provider in self.feature_providers:
            sample_count += provider.get_mode_size(mode)
        return sample_count

    def get_data(
        self,
        mode: str,
        batch_size: int,
        features_length: int,
        truncation_strategy: str = "default",
        augmentation_policy: dict = {
            "freq_mix_prob": 0.0,
            "time_mask_max_size": 0,
            "time_mask_count": 0,
            "freq_mask_max_size": 0,
            "freq_mask_count": 0,
        },
    ):
        """Gets spectrograms from the appropriate mode. Ensures spectrograms are the approriate length and optionally applies augmentation.

        Args:
            mode (str): which training set to count the spectrograms. One of `training`, `testing`, `testing_ambient`, `validation`, or `validation_ambient`
            batch_size (int): number of spectrograms in the sample for training mode
            features_length (int): the length of the spectrograms
            truncation_strategy (str): how to truncate spectrograms longer than `features_length`
            augmentation_policy (dict): dictionary that specifies augmentation settings. It has the following keys:
                freq_mix_prob: probability that FreqMix is applied
                time_mask_max_size: maximum size of time masks for SpecAugment
                time_mask_count: the total number of separate time masks applied for SpecAugment
                freq_mask_max_size: maximum size of frequency feature masks for SpecAugment
                freq_mask_count: the total number of separate feature masks applied for SpecAugment

        Returns:
            data: spectrograms in a NumPy array (or as a list if in mode is `*_ambient`)
            labels: ground truth for the spectrograms; i.e., whether a positive sample or negative sample
            weights: penalizing weight for incorrect predictions for each spectrogram
        """

        if mode == "training":
            sample_count = batch_size
        elif (mode == "validation") or (mode == "testing"):
            sample_count = self.get_mode_size(mode)

        data = []
        labels = []
        weights = []

        if mode == "training":
            random_feature_providers = random.choices(
                [
                    provider
                    for provider in self.feature_providers
                    if provider.get_mode_size("training")
                ],
                [
                    provider.sampling_weight
                    for provider in self.feature_providers
                    if provider.get_mode_size("training")
                ],
                k=sample_count,
            )

            for provider in random_feature_providers:
                spectrogram = provider.get_random_spectrogram(
                    "training", features_length, truncation_strategy
                )
                spectrogram = spec_augment(
                    spectrogram,
                    augmentation_policy["time_mask_max_size"],
                    augmentation_policy["time_mask_count"],
                    augmentation_policy["freq_mask_max_size"],
                    augmentation_policy["freq_mask_count"],
                )

                data.append(spectrogram)
                labels.append(int(provider.label))
                weights.append(float(provider.penalty_weight))
        else:
            for provider in self.feature_providers:
                generator = provider.get_feature_generator(
                    mode, features_length, truncation_strategy
                )

                for spectrogram in generator:
                    data.append(spectrogram)
                    labels.append(provider.label)
                    weights.append(provider.penalty_weight)

        if truncation_strategy != "none":
            # Spectrograms are all the same length, convert to numpy array
            data = np.array(data)
            labels = np.array(labels)
            weights = np.array(weights)

        if truncation_strategy == "none":
            # Spectrograms may be of different length
            return data, np.array(labels), np.array(weights)

        indices = np.arange(labels.shape[0])

        if mode in ("testing", "validation"):
            # Randomize the order of the data, weights, and labels
            np.random.shuffle(indices)
        

        return data[indices], labels[indices], weights[indices]
```

```inference.py
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

"""Functions and classes for using microwakeword models with audio files/data"""

# imports
from typing import Optional
import numpy as np
import tensorflow as tf
from speech_commands.audio.audio_utils import generate_features_for_clip


class Model:
    """
    Class for loading and running tflite microwakeword models

    Args:
        tflite_model_path (str): Path to tflite model file.
        stride (int | None, optional): Time dimension's stride. If None, then the stride is the input tensor's time dimension. Defaults to None.
    """

    def __init__(self, tflite_model_path: str, stride: Optional[int] = None):
        # Load tflite model
        interpreter = tf.lite.Interpreter(
            model_path=tflite_model_path,
        )
        interpreter.allocate_tensors()

        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()

        self.is_quantized_model = self.input_details[0]["dtype"] in (np.int8, np.uint8)
        self.input_feature_slices = self.input_details[0]["shape"][1]

        if stride is None:
            self.stride = self.input_feature_slices
        else:
            self.stride = stride

        for s in range(len(self.input_details)):
            if self.is_quantized_model:
                interpreter.set_tensor(
                    self.input_details[s]["index"],
                    np.zeros(self.input_details[s]["shape"], dtype=np.int8),
                )
            else:
                interpreter.set_tensor(
                    self.input_details[s]["index"],
                    np.zeros(self.input_details[s]["shape"], dtype=np.float32),
                )

        self.model = interpreter

    def predict_clip(self, data: np.ndarray, step_ms: int = 20):
        """Run the model on a single clip of audio data

        Args:
            data (numpy.ndarray): input data for the model (16 khz, 16-bit PCM audio data)
            step_ms (int): The window step sized used for generating the spectrogram in ms. Defaults to 20.

        Returns:
            list: model predictions for the input audio data
        """

        # Get the spectrogram
        spectrogram = generate_features_for_clip(data, step_ms=step_ms)

        return self.predict_spectrogram(spectrogram)

    def predict_spectrogram(self, spectrogram: np.ndarray):
        """Run the model on a single spectrogram

        Args:
            spectrogram (numpy.ndarray): Input spectrogram.

        Returns:
            list: model predictions for the input audio data
        """

        # Spectrograms with type np.uint16 haven't been scaled
        if np.issubdtype(spectrogram.dtype, np.uint16):
            spectrogram = spectrogram.astype(np.float32) * 0.0390625
        elif np.issubdtype(spectrogram.dtype, np.float64):
            spectrogram = spectrogram.astype(np.float32)

        # Slice the input data into the required number of chunks
        chunks = []
        for last_index in range(
            self.input_feature_slices, len(spectrogram) + 1, self.stride
        ):
            chunk = spectrogram[last_index - self.input_feature_slices : last_index]
            if len(chunk) == self.input_feature_slices:
                chunks.append(chunk)

        # Get the prediction for each chunk
        predictions = []
        for chunk in chunks:
            if self.is_quantized_model and spectrogram.dtype != np.int8:
                chunk = self.quantize_input_data(chunk, self.input_details[0])

            self.model.set_tensor(
                self.input_details[0]["index"],
                np.reshape(chunk, self.input_details[0]["shape"]),
            )
            self.model.invoke()

            raw = self.model.get_tensor(self.output_details[0]["index"])[0]  # shape: (num_classes,)

            # Descuantice si aplica (int8/uint8)
            out_dtype = self.output_details[0]["dtype"]
            if out_dtype in (np.uint8, np.int8):
                raw = self.dequantize_output_data(raw, self.output_details[0])

            # 'raw' debería ser probas (si la última capa es softmax)
            # Guarde el vector completo; si necesita una sola decisión, use argmax fuera
            predictions.append(raw)  # p.ej., predictions: List[np.ndarray(num_classes,)]

        return np.array(predictions, dtype=np.float32)

    def quantize_input_data(self, data: np.ndarray, input_details: dict) -> np.ndarray:
        """quantize the input data using scale and zero point

        Args:
            data (numpy.array in float): input data for the interpreter
            input_details (dict): output of get_input_details from the tflm interpreter.

        Returns:
          numpy.ndarray: quantized data as int8 dtype
        """
        # Get input quantization parameters
        data_type = input_details["dtype"]

        input_quantization_parameters = input_details["quantization_parameters"]
        input_scale, input_zero_point = (
            input_quantization_parameters["scales"][0],
            input_quantization_parameters["zero_points"][0],
        )
        # quantize the input data
        data = data / input_scale + input_zero_point
        return data.astype(data_type)

    def dequantize_output_data(
        self, data: np.ndarray, output_details: dict
    ) -> np.ndarray:
        """Dequantize the model output

        Args:
            data (numpy.ndarray): integer data to be dequantized
            output_details (dict): TFLM interpreter model output details

        Returns:
            numpy.ndarray: dequantized data as float32 dtype
        """
        output_quantization_parameters = output_details["quantization_parameters"]
        scales = output_quantization_parameters["scales"]
        zero_points = output_quantization_parameters["zero_points"]
        scale = float(scales[0]) if len(scales) else 1.0
        zero = int(zero_points[0]) if len(zero_points) else 0
        return (data.astype(np.float32) - zero) * scale
```

```mixednet.py
# coding=utf-8
# Copyright 2024 Kevin Ahrendt.
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

"""Model based on 1D depthwise MixedConvs and 1x1 convolutions in time + residual."""

from speech_commands.layers import stream
from speech_commands.layers import strided_drop

import ast
import tensorflow as tf


def parse(text):
    """Parse model parameters.

    Args:
      text: string with layer parameters: '128,128' or "'relu','relu'".

    Returns:
      list of parsed parameters
    """
    if not text:
        return []
    res = ast.literal_eval(text)
    if isinstance(res, tuple):
        return res
    else:
        return [res]


def model_parameters(parser_nn):
    """MixedNet model parameters."""

    parser_nn.add_argument(
        "--pointwise_filters",
        type=str,
        default="48, 48, 48, 48",
        help="Number of filters in every MixConv block's pointwise convolution",
    )
    parser_nn.add_argument(
        "--residual_connection",
        type=str,
        default="0,0,0,0,0",
        help="Use a residual connection in each MixConv block",
    )
    parser_nn.add_argument(
        "--repeat_in_block",
        type=str,
        default="1,1,1,1",
        help="Number of repeating conv blocks inside of residual block",
    )
    parser_nn.add_argument(
        "--mixconv_kernel_sizes",
        type=str,
        default="[5], [9], [13], [21]",
        help="Kernel size lists for DepthwiseConv1D in time dim for every MixConv block",
    )
    parser_nn.add_argument(
        "--max_pool",
        type=int,
        default=0,
        help="apply max pool instead of average pool before final convolution and sigmoid activation",
    )
    parser_nn.add_argument(
        "--first_conv_filters",
        type=int,
        default=32,
        help="Number of filters on initial convolution layer. Set to 0 to disable.",
    )
    parser_nn.add_argument(
        "--first_conv_kernel_size",
        type=int,
        default="3",
        help="Temporal kernel size for the initial convolution layer.",
    )
    parser_nn.add_argument(
        "--spatial_attention",
        type=int,
        default=0,
        help="Add a spatial attention layer before the final pooling layer",
    )
    parser_nn.add_argument(
        "--pooled",
        type=int,
        default=0,
        help="Pool the temporal dimension before the final fully connected layer. Uses average pooling or max pooling depending on the max_pool argument",
    )
    parser_nn.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Striding in the time dimension of the initial convolution layer",
    )


def spectrogram_slices_dropped(flags):
    """Computes the number of spectrogram slices dropped due to valid padding.

    Args:
        flags: data/model parameters

    Returns:
        int: number of spectrogram slices dropped
    """
    spectrogram_slices_dropped = 0

    if flags.first_conv_filters > 0:
        spectrogram_slices_dropped += flags.first_conv_kernel_size - 1

    for repeat, ksize in zip(
        parse(flags.repeat_in_block),
        parse(flags.mixconv_kernel_sizes),
    ):
        spectrogram_slices_dropped += (repeat * (max(ksize) - 1)) * flags.stride

    # spectrogram_slices_dropped *= flags.stride
    return spectrogram_slices_dropped


def _split_channels(total_filters, num_groups):
    """Helper for MixConv"""
    split = [total_filters // num_groups for _ in range(num_groups)]
    split[0] += total_filters - sum(split)
    return split


def _get_shape_value(maybe_v2_shape):
    """Helper for MixConv"""
    if maybe_v2_shape is None:
        return None
    elif isinstance(maybe_v2_shape, int):
        return maybe_v2_shape
    else:
        return maybe_v2_shape.value


class ChannelSplit(tf.keras.layers.Layer):
    def __init__(self, splits, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.splits = splits
        self.axis = axis

    def call(self, inputs):
        return tf.split(inputs, self.splits, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shapes = []
        for split in self.splits:
            new_shape = list(input_shape)
            new_shape[self.axis] = split
            output_shapes.append(tuple(new_shape))
        return output_shapes



class MixConv:
    """MixConv with mixed depthwise convolutional kernels.

    MDConv is an improved depthwise convolution that mixes multiple kernels (e.g.
    3x1, 5x1, etc). Right now, we use an naive implementation that split channels
    into multiple groups and perform different kernels for each group.

    See Mixnet paper for more details.
    """

    def __init__(self, kernel_size, **kwargs):
        """Initialize the layer.

        Most of args are the same as tf.keras.layers.DepthwiseConv2D.

        Args:
          kernel_size: An integer or a list. If it is a single integer, then it is
            same as the original tf.keras.layers.DepthwiseConv2D. If it is a list,
            then we split the channels and perform different kernel for each group.
          strides: An integer or tuple/list of 2 integers, specifying the strides of
            the convolution along the height and width.
          **kwargs: other parameters passed to the original depthwise_conv layer.
        """
        self._channel_axis = -1

        self.ring_buffer_length = max(kernel_size) - 1

        self.kernel_sizes = kernel_size

    def __call__(self, inputs):
        # We manually handle the streaming ring buffer for each layer
        #   - There is some latency overhead on the esp devices for loading each ring buffer's data
        #   - This avoids variable's holding redundant information
        #   - Reduces the necessary size of the tensor arena
        net = inputs

        if len(self.kernel_sizes) == 1:
            return tf.keras.layers.DepthwiseConv2D(
                (self.kernel_sizes[0], 1), strides=1, padding="valid"
            )(net)

        filters = _get_shape_value(net.shape[self._channel_axis])
        splits = _split_channels(filters, len(self.kernel_sizes))
        x_splits = ChannelSplit(splits, axis=self._channel_axis)(net)

        x_outputs = []
        for x, ks in zip(x_splits, self.kernel_sizes):
            fit = strided_drop.StridedKeep(ks)(x)
            x_outputs.append(
                tf.keras.layers.DepthwiseConv2D((ks, 1), strides=1, padding="valid")(
                    fit
                )
            )

        for i, output in enumerate(x_outputs):
            features_drop = output.shape[1] - x_outputs[-1].shape[1]
            x_outputs[i] = strided_drop.StridedDrop(features_drop)(output)

        x = tf.keras.layers.concatenate(x_outputs, axis=self._channel_axis)
        return x


def model(flags, shape, batch_size, num_classes=2):
    """MixedNet model.

    It is based on the paper
    MixConv: Mixed Depthwise Convolutional Kernels
    https://arxiv.org/abs/1907.09595
    Args:
      flags: data/model parameters
      shape: shape of the input vector
      config: dictionary containing microWakeWord training configuration

    Returns:
      Keras model for training
    """

    pointwise_filters = parse(flags.pointwise_filters)
    repeat_in_block = parse(flags.repeat_in_block)
    mixconv_kernel_sizes = parse(flags.mixconv_kernel_sizes)
    residual_connections = parse(flags.residual_connection)

    for list in (
        pointwise_filters,
        repeat_in_block,
        mixconv_kernel_sizes,
        residual_connections,
    ):
        if len(pointwise_filters) != len(list):
            raise ValueError("all input lists have to be the same length")

    input_audio = tf.keras.layers.Input(
        shape=shape,
        batch_size=batch_size,
    )
    net = input_audio

    # make it [batch, time, 1, feature]
    net = tf.keras.ops.expand_dims(net, axis=2)

    # encoder
    for filters, repeat, ksize, res in zip(
        pointwise_filters,
        repeat_in_block,
        mixconv_kernel_sizes,
        residual_connections,
    ):
        if res:
            residual = tf.keras.layers.Conv2D(
                filters=filters, kernel_size=1, use_bias=False, padding="same"
            )(net)
            residual = tf.keras.layers.BatchNormalization()(residual)

        for _ in range(repeat):
            if max(ksize) > 1:
                net = MixConv(kernel_size=ksize)(net)
            net = tf.keras.layers.Conv2D(
                filters=filters, kernel_size=1, use_bias=False, padding="same"
            )(net)
            net = tf.keras.layers.BatchNormalization()(net)

            if res:
                residual = strided_drop.StridedDrop(residual.shape[1] - net.shape[1])(
                    residual
                )
                net = net + residual

            net = tf.keras.layers.Activation("relu")(net)

    if net.shape[1] > 1:
        if flags.pooled:
            # We want to use either Global Max Pooling or Global Average Pooling, but the esp-nn operator optimizations only benefit regular pooling operations

            if flags.max_pool:
                net = tf.keras.layers.MaxPooling2D(pool_size=(net.shape[1], 1))(net)
            else:
                net = tf.keras.layers.AveragePooling2D(pool_size=(net.shape[1], 1))(net)

    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(num_classes, activation="softmax")(net)

    return tf.keras.Model(input_audio, net)
```

```test.py
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
```

```train.py
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
```

```utils.py
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

"""Utility functions for operations on Model."""
import os.path
import numpy as np
import tensorflow as tf

from absl import logging

from speech_commands.layers import modes, stream, strided_drop


def _set_mode(model, mode):
    """Set model's inference type and disable training."""

    def _recursive_set_layer_mode(layer, mode):
        if isinstance(layer, tf.keras.layers.Wrapper):
            _recursive_set_layer_mode(layer.layer, mode)

        config = layer.get_config()
        # for every layer set mode, if it has it
        if "mode" in config:
            assert isinstance(
                layer,
                (stream.Stream, strided_drop.StridedDrop, strided_drop.StridedKeep),
            )
            layer.mode = mode
        # with any mode of inference - training is False
        if "training" in config:
            layer.training = False
        if mode == modes.Modes.NON_STREAM_INFERENCE:
            if "unroll" in config:
                layer.unroll = True

    for layer in model.layers:
        _recursive_set_layer_mode(layer, mode)
    return model


def _copy_weights(new_model, model):
    """Copy weights of trained model to an inference one."""

    def _same_weights(weight, new_weight):
        # Check that weights are the same
        # Note that states should be marked as non trainable
        return (
            weight.trainable == new_weight.trainable
            and weight.shape == new_weight.shape
            and weight.name[weight.name.rfind("/") : None]
            == new_weight.name[new_weight.name.rfind("/") : None]
        )

    if len(new_model.layers) != len(model.layers):
        raise ValueError(
            "number of layers in new_model: %d != to layers number in model: %d "
            % (len(new_model.layers), len(model.layers))
        )

    for i in range(len(model.layers)):
        layer = model.layers[i]
        new_layer = new_model.layers[i]

        # if number of weights in the layers are the same
        # then we can set weights directly
        if len(layer.get_weights()) == len(new_layer.get_weights()):
            new_layer.set_weights(layer.get_weights())
        elif layer.weights:
            k = 0  # index pointing to weights in the copied model
            new_weights = []
            # iterate over weights in the new_model
            # and prepare a new_weights list which will
            # contain weights from model and weight states from new model
            for k_new in range(len(new_layer.get_weights())):
                new_weight = new_layer.weights[k_new]
                new_weight_values = new_layer.get_weights()[k_new]
                same_weights = True

                # if there are weights which are not copied yet
                if k < len(layer.get_weights()):
                    weight = layer.weights[k]
                    weight_values = layer.get_weights()[k]
                    if (
                        weight.shape != weight_values.shape
                        or new_weight.shape != new_weight_values.shape
                    ):
                        raise ValueError("weights are not listed in order")

                    # if there are weights available for copying and they are the same
                    if _same_weights(weight, new_weight):
                        new_weights.append(weight_values)
                        k = k + 1  # go to next weight in model
                    else:
                        same_weights = False  # weights are different
                else:
                    same_weights = (
                        False  # all weights are copied, remaining is different
                    )

                if not same_weights:
                    # weight with index k_new is missing in model,
                    # so we will keep iterating over k_new until find similar weights
                    new_weights.append(new_weight_values)

            # check that all weights from model are copied to a new_model
            if k != len(layer.get_weights()):
                raise ValueError(
                    "trained model has: %d weights, but only %d were copied"
                    % (len(layer.get_weights()), k)
                )

            # now they should have the same number of weights with matched sizes
            # so we can set weights directly
            new_layer.set_weights(new_weights)
    return new_model


def save_model_summary(model, path, file_name="model_summary.txt"):
    """Saves model topology/summary in text format.

    Args:
      model: Keras model
      path: path where to store model summary
      file_name: model summary file name
    """
    with tf.io.gfile.GFile(os.path.join(path, file_name), "w") as fd:
        stringlist = []
        model.summary(
            print_fn=lambda x: stringlist.append(x)
        )  # pylint: disable=unnecessary-lambda
        model_summary = "\n".join(stringlist)
        fd.write(model_summary)


def convert_to_inference_model(model, input_tensors, mode):
    """Convert tf._keras_internal.engine.functional `Model` instance to a streaming inference.

    It will create a new model with new inputs: input_tensors.
    All weights will be copied. Internal states for streaming mode will be created
    Only tf._keras_internal.engine.functional Keras model is supported!

    Args:
        model: Instance of `Model`.
        input_tensors: list of input tensors to build the model upon.
        mode: is defined by modes.Modes

    Returns:
        An instance of streaming inference `Model` reproducing the behavior
        of the original model, on top of new inputs tensors,
        using copied weights.

    Raises:
        ValueError: in case of invalid `model` argument value or input_tensors
    """

    # scope is introduced for simplifiyng access to weights by names
    scope_name = "streaming"

    with tf.name_scope(scope_name):
        if not isinstance(model, tf.keras.Model):
            raise ValueError(
                "Expected `model` argument to be a `Model` instance, got ", model
            )
        if isinstance(model, tf.keras.Sequential):
            raise ValueError(
                "Expected `model` argument "
                "to be a functional `Model` instance, "
                "got a `Sequential` instance instead:",
                model,
            )
        model = _set_mode(model, mode)
        new_model = tf.keras.models.clone_model(model, input_tensors)

    if mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
        return _copy_weights(new_model, model)
    elif mode == modes.Modes.NON_STREAM_INFERENCE:
        new_model.set_weights(model.get_weights())
        return new_model
    else:
        raise ValueError("non supported mode ", mode)


def to_streaming_inference(model_non_stream, config, mode):
    """Convert non streaming trained model to inference modes.

    Args:
      model_non_stream: trained Keras model non streamable
      config: dictionary containing microWakeWord training configuration
      mode: it supports Non streaming inference or Streaming inference with internal
        states

    Returns:
      Keras inference model of inference_type
    """

    input_data_shape = modes.get_input_data_shape(config, mode)

    # get input data type and use it for input streaming type
    if isinstance(model_non_stream.input, (tuple, list)):
        dtype = model_non_stream.input[0].dtype
    else:
        dtype = model_non_stream.input.dtype

    # For streaming, set the batch size to 1
    input_tensors = [
        tf.keras.layers.Input(
            shape=input_data_shape, batch_size=1, dtype=dtype, name="input_audio"
        )
    ]

    if (
        isinstance(model_non_stream.input, (tuple, list))
        and len(model_non_stream.input) > 1
    ):
        if len(model_non_stream.input) > 2:
            raise ValueError(
                "Maximum number of inputs supported is 2 (input_audio and "
                "cond_features), but got %d inputs" % len(model_non_stream.input)
            )

        input_tensors.append(
            tf.keras.layers.Input(
                shape=config["cond_shape"],
                batch_size=1,
                dtype=model_non_stream.input[1].dtype,
                name="cond_features",
            )
        )

    # Input tensors must have the same shape as the original
    if isinstance(model_non_stream.input, (tuple, list)):
        model_inference = convert_to_inference_model(
            model_non_stream, input_tensors, mode
        )
    else:
        model_inference = convert_to_inference_model(
            model_non_stream, input_tensors[0], mode
        )

    return model_inference


def model_to_saved(
    model_non_stream,
    config,
    mode=modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
):
    """Convert Keras model to SavedModel.

    Depending on mode:
      1 Converted inference graph and model will be streaming statefull.
      2 Converted inference graph and model will be non streaming stateless.

    Args:
      model_non_stream: Keras non streamable model
      config: dictionary containing microWakeWord training configuration
      mode: inference mode it can be streaming with internal state or non
        streaming
    """

    if mode not in (
        modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
        modes.Modes.NON_STREAM_INFERENCE,
    ):
        raise ValueError("mode %s is not supported " % mode)

    if mode == modes.Modes.NON_STREAM_INFERENCE:
        model = model_non_stream
    else:
        # convert non streaming Keras model to Keras streaming model, internal state
        model = to_streaming_inference(model_non_stream, config, mode)

    return model


def convert_saved_model_to_tflite(
    config, audio_processor, path_to_model, folder, fname, quantize=False
):
    """Convert SavedModel to TFLite and optionally quantize it.

    Args:
        config: dictionary containing microWakeWord training configuration
        audio_processor:  microWakeWord FeatureHandler object for retrieving spectrograms
        path_to_model: path to SavedModel
        folder: folder where converted model will be saved
        fname: output filename for TFLite file
        quantize: boolean selecting whether to quantize the model
    """

    def representative_dataset_gen():
        sample_fingerprints, _, _ = audio_processor.get_data(
            "training", 500, features_length=config["spectrogram_length"]
        )

        sample_fingerprints[0][
            0, 0
        ] = 0.0  # guarantee one pixel is the preprocessor min
        sample_fingerprints[0][
            0, 1
        ] = 26.0  # guarantee one pixel is the preprocessor max

        for spectrogram in sample_fingerprints:
            # Asegúrate de expandir dimensión batch si falta
            spectrogram = np.expand_dims(spectrogram, axis=0).astype(np.float32)
            yield [spectrogram]

        # stride = config["stride"]

        # for spectrogram in sample_fingerprints:
        #     assert spectrogram.shape[0] % stride == 0

        #     for i in range(0, spectrogram.shape[0] - stride, stride):
        #         sample = spectrogram[i : i + stride, :].astype(np.float32)
        #         yield [sample]

    converter = tf.lite.TFLiteConverter.from_saved_model(path_to_model)
    converter.optimizations = {tf.lite.Optimize.DEFAULT}

    # Without this flag, the Streaming layer `state` variables are left as float32,
    # resulting in Quantize and Dequantize operations before and after every `ReadVariable`
    # and `AssignVariable` operation.
    converter._experimental_variable_quantization = True

    if quantize:
        converter.target_spec.supported_ops = {tf.lite.OpsSet.TFLITE_BUILTINS_INT8}
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.uint8
        converter.representative_dataset = tf.lite.RepresentativeDataset(
            representative_dataset_gen
        )

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(os.path.join(folder, fname), "wb") as f:
        tflite_model = converter.convert()
        f.write(tflite_model)


def convert_model_saved(model, config, folder, mode):
    """Convert model to streaming and non streaming SavedModel.

    Args:
        model: model settings
        config: dictionary containing microWakeWord training configuration
        folder: folder where converted model will be saved
        mode: inference mode
    """

    path_model = os.path.join(config["train_dir"], folder)
    if not os.path.exists(path_model):
        os.makedirs(path_model)

    # Convert trained model to SavedModel
    converted_model = model_to_saved(model, config, mode)
    converted_model.summary()

    assert converted_model.input.shape[0] is not None

    # XXX: Using `converted_model.export(path_model)` results in obscure errors during
    # quantization, we create an export archive directly instead.
    export_archive = tf.keras.export.ExportArchive()
    export_archive.track(converted_model)
    export_archive.add_endpoint(
        name="serve",
        fn=converted_model.call,
        input_signature=[tf.TensorSpec(shape=converted_model.input.shape, dtype=tf.float32)],
    )
    export_archive.write_out(path_model)

    save_model_summary(converted_model, path_model)

    return converted_model
```

```audio_utils.py
# coding=utf-8
# Copyright 2024 Kevin Ahrendt.
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

import numpy as np
import tensorflow as tf
import webrtcvad

from tensorflow.lite.experimental.microfrontend.python.ops import (
    audio_microfrontend_op as frontend_op,
)
from scipy.io import wavfile

from pymicro_features import MicroFrontend


def generate_features_for_clip(
    audio_samples: np.ndarray, step_ms: int = 20, use_c: bool = True
):
    """Generates spectrogram features for the given audio data.

    Args:
        audio_samples (numpy.ndarray): The clip's audio samples.
        step_ms (int, optional): The window step size in ms. Defaults to 20.
        use_c (bool, optional): Whether to use the C implementation of the microfrontend via pymicro-features. Defaults to True.

    Raises:
        ValueError: If the provided audio data is not a 16-bit integer array.


    Returns:
        numpy.ndarray: The spectrogram features for the provided audio clip.
    """

    # Convert any float formatted audio data to an int16 array
    if audio_samples.dtype in (np.float32, np.float64):
        audio_samples = np.clip((audio_samples * 32768), -32768, 32767).astype(np.int16)

    if use_c:
        audio_samples = audio_samples.tobytes()
        micro_frontend = MicroFrontend()
        features = []
        audio_idx = 0
        num_audio_bytes = len(audio_samples)
        while audio_idx + 160 * 2 < num_audio_bytes:
            frontend_result = micro_frontend.ProcessSamples(
                audio_samples[audio_idx : audio_idx + 160 * 2]
            )
            audio_idx += frontend_result.samples_read * 2
            if frontend_result.features:
                features.append(frontend_result.features)

        return np.array(features).astype(np.float32)

    with tf.device("/cpu:0"):
        # The default settings match the TFLM preprocessor settings.
        # Preproccesor model is available from the tflite-micro repository, accessed December 2023.
        micro_frontend = frontend_op.audio_microfrontend(
            tf.convert_to_tensor(audio_samples),
            sample_rate=16000,
            window_size=30,
            window_step=step_ms,
            num_channels=40,
            upper_band_limit=7500,
            lower_band_limit=125,
            enable_pcan=True,
            min_signal_remaining=0.05,
            out_scale=1,
            out_type=tf.uint16,
        )

        spectrogram = micro_frontend.numpy()
        return spectrogram


def save_clip(audio_samples: np.ndarray, output_file: str) -> None:
    """Saves an audio clip's sample as a wave file.

    Args:
        audio_samples (numpy.ndarray): The clip's audio samples.
        output_file (str): Path to the desired output file.
    """
    if audio_samples.dtype in (np.float32, np.float64):
        audio_samples = (audio_samples * 32767).astype(np.int16)
    wavfile.write(output_file, 16000, audio_samples)


def remove_silence_webrtc(
    audio_data: np.ndarray,
    frame_duration: float = 0.030,
    sample_rate: int = 16000,
    min_start: int = 2000,
) -> np.ndarray:
    """Uses webrtc voice activity detection to remove silence from the clips

    Args:
        audio_data (numpy.ndarray): The input clip's audio samples.
        frame_duration (float): The frame_duration for webrtcvad. Defaults to 0.03.
        sample_rate (int): The audio's sample rate. Defaults to 16000.
        min_start: (int): The number of audio samples from the start of the clip to always include. Defaults to 2000.

    Returns:
        numpy.ndarray: Array with the trimmed audio clip's samples.
    """
    vad = webrtcvad.Vad(0)

    # webrtcvad expects int16 arrays as input, so convert if audio_data is a float
    float_type = audio_data.dtype in (np.float32, np.float64)
    if float_type:
        audio_data = (audio_data * 32767).astype(np.int16)

    filtered_audio = audio_data[0:min_start].tolist()

    step_size = int(sample_rate * frame_duration)

    for i in range(min_start, audio_data.shape[0] - step_size, step_size):
        vad_detected = vad.is_speech(
            audio_data[i : i + step_size].tobytes(), sample_rate
        )
        if vad_detected:
            # If voice activity is detected, add it to filtered_audio
            filtered_audio.extend(audio_data[i : i + step_size].tolist())

    # If the original audio data was a float array, convert back
    if float_type:
        trimmed_audio = np.array(filtered_audio)
        return np.array(trimmed_audio / 32767).astype(np.float32)

    return np.array(filtered_audio).astype(np.int16)```

```augmentation.py
# coding=utf-8
# Copyright 2024 Kevin Ahrendt.
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

import audiomentations
import warnings

import numpy as np

from typing import List, Optional


class Augmentation:
    """A class that handles applying augmentations to audio clips.

    Args:
        augmentation_duration_s (float): The duration of the augmented clip in seconds.
        augmentation_probabilities (dict, optional): Dictionary that specifies each augmentation's probability of being applied. Defaults to { "SevenBandParametricEQ": 0.0, "TanhDistortion": 0.0, "PitchShift": 0.0, "BandStopFilter": 0.0, "AddColorNoise": 0.25, "AddBackgroundNoise": 0.75, "Gain": 1.0, "GainTransition": 0.25, "RIR": 0.5, }.
        impulse_paths (List[str], optional): List of directory paths that contain room impulse responses that the audio clip is reverberated with. If the list is empty, then reverberation is not applied. Defaults to [].
        background_paths (List[str], optional): List of directory paths that contain audio clips to be mixed into the audio clip. If the list is empty, then the background augmentation is not applied. Defaults to [].
        background_min_snr_db (int, optional): The minimum signal to noise ratio for mixing in background audio. Defaults to -10.
        background_max_snr_db (int, optional): The maximum signal to noise ratio for mixing in background audio. Defaults to 10.
        min_gain_db (float, optional): The minimum gain for the gain augmentation. Defaults to -45.0.
        max_gain_db (float, optional): The mmaximum gain for the gain augmentation. Defaults to 0.0.
        min_gain_transition_db (float, optional): The minimum gain for the gain transition augmentation. Defaults to -10.0.
        max_gain_transition_db (float, optional): The mmaximum gain for the gain transition augmentation. Defaults to 10.0.
        min_jitter_s (float, optional): The minimum duration in seconds that the original clip is positioned before the end of the augmented audio. Defaults to 0.0.
        max_jitter_s (float, optional): The maximum duration in seconds that the original clip is positioned before the end of the augmented audio. Defaults to 0.0.
        truncate_randomly: (bool, option): If true, the clip is truncated to the specified duration randomly. Otherwise, the start of the clip is truncated.
    """

    def __init__(
        self,
        augmentation_duration_s: Optional[float] = None,
        augmentation_probabilities: dict = {
            "SevenBandParametricEQ": 0.0,
            "TanhDistortion": 0.0,
            "PitchShift": 0.0,
            "BandStopFilter": 0.0,
            "AddColorNoise": 0.25,
            "AddBackgroundNoise": 0.75,
            "Gain": 1.0,
            "GainTransition": 0.25,
            "RIR": 0.5,
        },
        impulse_paths: List[str] = [],
        background_paths: List[str] = [],
        background_min_snr_db: int = -10,
        background_max_snr_db: int = 10,
        color_min_snr_db: int = 10,
        color_max_snr_db: int = 30,
        min_gain_db: float = -45,
        max_gain_db: float = 0,
        min_gain_transition_db: float = -10,
        max_gain_transition_db: float = 10,
        min_jitter_s: float = 0.0,
        max_jitter_s: float = 0.0,
        truncate_randomly: bool = False,
    ):
        self.truncate_randomly = truncate_randomly
        ############################################
        # Configure audio duration and positioning #
        ############################################

        self.min_jitter_samples = int(min_jitter_s * 16000)
        self.max_jitter_samples = int(max_jitter_s * 16000)

        if augmentation_duration_s is not None:
            self.augmented_samples = int(augmentation_duration_s * 16000)
        else:
            self.augmented_samples = None

        assert (
            self.min_jitter_samples <= self.max_jitter_samples
        ), "Minimum jitter must be less than or equal to maximum jitter."

        #######################
        # Setup augmentations #
        #######################

        # If either the background_paths or impulse_paths are not specified, use an identity transform instead
        def identity_transform(samples, sample_rate):
            return samples

        background_noise_augment = audiomentations.Lambda(
            transform=identity_transform, p=0.0
        )
        reverb_augment = audiomentations.Lambda(transform=identity_transform, p=0.0)

        if len(background_paths):
            background_noise_augment = audiomentations.AddBackgroundNoise(
                p=augmentation_probabilities.get("AddBackgroundNoise", 0.0),
                sounds_path=background_paths,
                min_snr_db=background_min_snr_db,
                max_snr_db=background_max_snr_db,
            )

        if len(impulse_paths) > 0:
            reverb_augment = audiomentations.ApplyImpulseResponse(
                p=augmentation_probabilities.get("RIR", 0.0),
                ir_path=impulse_paths,
            )

        # Based on openWakeWord's augmentations, accessed on February 23, 2024.
        self.augment = audiomentations.Compose(
            transforms=[
                audiomentations.SevenBandParametricEQ(
                    p=augmentation_probabilities.get("SevenBandParametricEQ", 0.0),
                    min_gain_db=-6,
                    max_gain_db=6,
                ),
                audiomentations.TanhDistortion(
                    p=augmentation_probabilities.get("TanhDistortion", 0.0),
                    min_distortion=0.0001,
                    max_distortion=0.10,
                ),
                audiomentations.PitchShift(
                    p=augmentation_probabilities.get("PitchShift", 0.0),
                    min_semitones=-3,
                    max_semitones=3,
                ),
                audiomentations.BandStopFilter(
                    p=augmentation_probabilities.get("BandStopFilter", 0.0),
                ),
                audiomentations.AddColorNoise(
                    p=augmentation_probabilities.get("AddColorNoise", 0.0),
                    min_snr_db=color_min_snr_db,
                    max_snr_db=color_max_snr_db,
                ),
                background_noise_augment,
                audiomentations.Gain(
                    p=augmentation_probabilities.get("Gain", 0.0),
                    min_gain_db=min_gain_db,
                    max_gain_db=max_gain_db,
                ),
                audiomentations.GainTransition(
                    p=augmentation_probabilities.get("GainTransition", 0.0),
                    min_gain_db=min_gain_transition_db,
                    max_gain_db=max_gain_transition_db,
                ),
                reverb_augment,
                audiomentations.Compose(
                    transforms=[
                        audiomentations.Normalize(
                            apply_to="only_too_loud_sounds", p=1.0
                        )
                    ]
                ),  # If the audio is clipped, normalize
            ],
            shuffle=False,
        )

    def add_jitter(self, input_audio: np.ndarray):
        """Pads the clip on the right by a random duration between the class's min_jitter_s and max_jitter_s paramters.

        Args:
            input_audio (numpy.ndarray): Array containing the audio clip's samples.

        Returns:
            numpy.ndarray: Array of audio samples with silence added to the end.
        """
        if self.min_jitter_samples < self.max_jitter_samples:
            jitter_samples = np.random.randint(
                self.min_jitter_samples, self.max_jitter_samples
            )
        else:
            jitter_samples = self.min_jitter_samples

        # Pad audio on the right by jitter samples
        return np.pad(input_audio, (0, jitter_samples))

    def create_fixed_size_clip(self, input_audio: np.ndarray):
        """Ensures the input audio clip has a fixced length. If the duration is too long, the start of the clip is removed. If it is too short, the start of the clip is padded with silence.

        Args:
            input_audio (numpy.ndarray): Array containing the audio clip's samples.

        Returns:
            numpy.ndarray: Array of audio samples with `augmented_duration_s` length.
        """
        if self.augmented_samples is None:
            return input_audio

        if self.augmented_samples < input_audio.shape[0]:
            # Truncate the too long audio by removing the start of the clip
            if self.truncate_randomly:
                random_start = np.random.randint(
                    0, input_audio.shape[0] - self.augmented_samples
                )
                input_audio = input_audio[
                    random_start : random_start + self.augmented_samples
                ]
            else:
                input_audio = input_audio[-self.augmented_samples :]
        else:
            # Pad with zeros at start of too short audio clip
            left_padding_samples = self.augmented_samples - input_audio.shape[0]

            input_audio = np.pad(input_audio, (left_padding_samples, 0))

        return input_audio

    def augment_clip(self, input_audio: np.ndarray):
        """Augments the input audio after adding jitter and creating a fixed size clip.

        Args:
            input_audio (numpy.ndarray): Array containing the audio clip's samples.

        Returns:
            numpy.ndarray: The augmented audio of fixed duration.
        """
        input_audio = self.add_jitter(input_audio)
        input_audio = self.create_fixed_size_clip(input_audio)

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )  # Suppresses warning about background clip being too quiet... TODO: find better approach!
            output_audio = self.augment(input_audio, sample_rate=16000)

        return output_audio

    def augment_generator(self, audio_generator):
        """A Python generator that augments clips retrived from the input audio generator.

        Args:
            audio_generator (generator): A Python generator that yields audio clips.

        Yields:
            numpy.ndarray: The augmented audio clip's samples.
        """
        for audio in audio_generator:
            yield self.augment_clip(audio)
```

```clips.py
# coding=utf-8
# Copyright 2024 Kevin Ahrendt.
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

from typing import Optional, Union
import audio_metadata
import datasets
import math
import os
import random
import wave

import numpy as np

from pathlib import Path

from speech_commands.audio.audio_utils import remove_silence_webrtc


class Clips:
    """Class for loading audio clips from the specified directory. The clips can first be filtered by their duration using the `min_clip_duration_s` and `max_clip_duration_s` parameters. Clips are retrieved as numpy float arrays via the `get_random_clip` method or via the `audio_generator` or `random_audio_generator` generators. Before retrieval, the audio clip can trim non-voice activiity. Before retrieval, the audio clip can be repeated until it is longer than a specified minimum duration.

    Args:
        input_directory (str): Path to audio clip files.
        file_pattern (str): File glob pattern for selecting audio clip files.
        min_clip_duration_s (float | None, optional): The minimum clip duration (in seconds). Set to None to disable filtering by minimum clip duration. Defaults to None.
        max_clip_duration_s (float | None, optional): The maximum clip duration (in seconds). Set to None to disable filtering by maximum clip duration. Defaults to None.
        repeat_clip_min_duration_s (float | None, optional): If a clip is shorter than this duration, then it is repeated until it is longer than this duration. Set to None to disable repeating the clip. Defaults to None.
        remove_silence (bool, optional): Use webrtcvad to trim non-voice activity in the clip. Defaults to False.
        random_split_seed (int | None, optional): The random seed used to split the clips into different sets. Set to None to disable splitting the clips. Defaults to None.
        split_count (int | float, optional): The percentage/count of clips to be included in the testing and validation sets. Defaults to 0.1.
        trimmed_clip_duration_s: (float | None, optional): The duration of the clips to trim the end of long clips. Set to None to disable trimming. Defaults to None.
        trim_zerios: (bool, optional): If true, any leading and trailling zeros are removed. Defaults to false.
    """

    def __init__(
        self,
        input_directory: str,
        file_pattern: str,
        min_clip_duration_s: Optional[float] = None,
        max_clip_duration_s: Optional[float] = None,
        repeat_clip_min_duration_s: Optional[float] = None,
        remove_silence: bool = False,
        random_split_seed: Optional[int] = None,
        split_count: Union[int, float] = 0.1,
        trimmed_clip_duration_s: Optional[float] = None,
        trim_zeros: bool = False,
    ):
        self.trim_zeros = trim_zeros
        self.trimmed_clip_duration_s = trimmed_clip_duration_s

        if min_clip_duration_s is not None:
            self.min_clip_duration_s = min_clip_duration_s
        else:
            self.min_clip_duration_s = 0.0

        if max_clip_duration_s is not None:
            self.max_clip_duration_s = max_clip_duration_s
        else:
            self.max_clip_duration_s = math.inf

        if repeat_clip_min_duration_s is not None:
            self.repeat_clip_min_duration_s = repeat_clip_min_duration_s
        else:
            self.repeat_clip_min_duration_s = 0.0

        self.remove_silence = remove_silence

        self.remove_silence_function = remove_silence_webrtc

        paths_to_clips = [str(i) for i in Path(input_directory).glob(file_pattern)]

        if (self.min_clip_duration_s == 0) and (math.isinf(self.max_clip_duration_s)):
            # No durations specified, so do not filter by length
            filtered_paths = paths_to_clips
        else:
            # Filter audio clips by length
            if file_pattern.endswith("wav"):
                # If it is a wave file, assume all wave files have the same parameters and filter by file size.
                # Based on openWakeWord's estimate_clip_duration and filter_audio_paths in data.py, accessed March 2, 2024.
                with wave.open(paths_to_clips[0], "rb") as input_wav:
                    channels = input_wav.getnchannels()
                    sample_width = input_wav.getsampwidth()
                    sample_rate = input_wav.getframerate()
                    frames = input_wav.getnframes()

                sizes = []
                sizes.extend([os.path.getsize(i) for i in paths_to_clips])

                # Correct for the wav file header bytes. Assumes all files in the directory have same parameters.
                header_correction = (
                    os.path.getsize(paths_to_clips[0])
                    - frames * sample_width * channels
                )

                durations = []
                for size in sizes:
                    durations.append(
                        (size - header_correction)
                        / (sample_rate * sample_width * channels)
                    )

                filtered_paths = [
                    path_to_clip
                    for path_to_clip, duration in zip(paths_to_clips, durations)
                    if (self.min_clip_duration_s < duration)
                    and (duration < self.max_clip_duration_s)
                ]
            else:
                # If not a wave file, use the audio_metadata package to analyze audio file headers for the duration.
                # This is slower!
                filtered_paths = []

                if (self.min_clip_duration_s > 0) or (
                    not math.isinf(self.max_clip_duration_s)
                ):
                    for audio_file in paths_to_clips:
                        metadata = audio_metadata.load(audio_file)
                        duration = metadata["streaminfo"]["duration"]
                        if (self.min_clip_duration_s < duration) and (
                            duration < self.max_clip_duration_s
                        ):
                            filtered_paths.append(audio_file)

        # Load all filtered clips
        audio_dataset = datasets.Dataset.from_dict(
            {"audio": [str(i) for i in filtered_paths]}
        ).cast_column("audio", datasets.Audio())

        # Convert all clips to 16 kHz sampling rate when accessed
        audio_dataset = audio_dataset.cast_column(
            "audio", datasets.Audio(sampling_rate=16000)
        )

        if random_split_seed is not None:
            train_testvalid = audio_dataset.train_test_split(
                test_size=2 * split_count, seed=random_split_seed
            )
            test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
            split_dataset = datasets.DatasetDict(
                {
                    "train": train_testvalid["train"],
                    "test": test_valid["test"],
                    "validation": test_valid["train"],
                }
            )
            self.split_clips = split_dataset

        self.clips = audio_dataset

    def audio_generator(self, split: Optional[str] = None, repeat: int = 1):
        """A Python generator that retrieves all loaded audio clips.

        Args:
            split (str | None, optional): Specifies which set the clips are retrieved from. If None, all clips are retrieved. Otherwise, it can be set to `train`, `test`, or `validation`. Defaults to None.
            repeat (int, optional): The number of times each audio clip will be yielded. Defaults to 1.

        Yields:
            numpy.ndarray: Array with the audio clip's samples.
        """
        if split is None:
            clip_list = self.clips
        else:
            clip_list = self.split_clips[split]
        for _ in range(repeat):
            for clip in clip_list:
                clip_audio = clip["audio"]["array"]

                if self.remove_silence:
                    clip_audio = self.remove_silence_function(clip_audio)

                if self.trim_zeros:
                    clip_audio = np.trim_zeros(clip_audio)

                if self.trimmed_clip_duration_s:
                    total_samples = int(self.trimmed_clip_duration_s * 16000)
                    clip_audio = clip_audio[:total_samples]

                clip_audio = self.repeat_clip(clip_audio)
                yield clip_audio

    def get_random_clip(self):
        """Retrieves a random audio clip.

        Returns:
            numpy.ndarray: Array with the audio clip's samples.
        """
        rand_audio_entry = random.choice(self.clips)
        clip_audio = rand_audio_entry["audio"]["array"]

        if self.remove_silence:
            clip_audio = self.remove_silence_function(clip_audio)

        if self.trim_zeros:
            clip_audio = np.trim_zeros(clip_audio)

        if self.trimmed_clip_duration_s:
            total_samples = int(self.trimmed_clip_duration_s * 16000)
            clip_audio = clip_audio[:total_samples]

        clip_audio = self.repeat_clip(clip_audio)
        return clip_audio

    def random_audio_generator(self, max_clips: int = math.inf):
        """A Python generator that retrieves random audio clips.

        Args:
            max_clips (int, optional): The total number of clips the generator will yield before the StopIteration. Defaults to math.inf.

        Yields:
            numpy.ndarray: Array with the random audio clip's samples.
        """
        while max_clips > 0:
            max_clips -= 1

            yield self.get_random_clip()

    def repeat_clip(self, audio_samples: np.array):
        """Repeats the audio clip until its duration exceeds the minimum specified in the class.

        Args:
            audio_samples numpy.ndarray: Original audio clip's samples.

        Returns:
            numpy.ndarray: Array with duration exceeding self.repeat_clip_min_duration_s.
        """
        original_clip = audio_samples
        desired_samples = int(self.repeat_clip_min_duration_s * 16000)
        while audio_samples.shape[0] < desired_samples:
            audio_samples = np.append(audio_samples, original_clip)
        return audio_samples
```

```spectrograms.py
# coding=utf-8
# Copyright 2024 Kevin Ahrendt.
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

from typing import Optional
import numpy as np

from speech_commands.audio.audio_utils import generate_features_for_clip
from speech_commands.audio.augmentation import Augmentation
from speech_commands.audio.clips import Clips


class SpectrogramGeneration:
    """A class that handles generating spectrogram features for audio clips. Spectrograms can optionally be split into nonoverlapping segments for faster file loading or they can optionally be strided by dropping the last feature windows to simulate a streaming model's sequential inputs.

    Args:
        clips (Clips): Object that retrieves audio clips.
        augmenter (Augmentation | None, optional): Object that augments audio clips. If None, no augmentations are applied. Defaults to None.
        step_ms (int, optional): The window step size in ms for the spectrogram features. Defaults to 20.
        split_spectrogram_duration_s (float | None, optional): Splits generated spectrograms to yield nonoverlapping spectrograms with this duration. If None, the entire spectrogram is yielded. Defaults to None.
        slide_frames (int | None, optional): Strides the generated spectrograms to yield `slide_frames` overlapping spectrogram by removing features at the end of the spectrogram. If None, the entire spectrogram is yielded. Defaults to None.
    """

    def __init__(
        self,
        clips: Clips,
        augmenter: Optional[Augmentation] = None,
        step_ms: int = 20,
        split_spectrogram_duration_s: Optional[float] = None,
        slide_frames: Optional[int] = None,
    ):

        self.clips = clips
        self.augmenter = augmenter
        self.step_ms = step_ms
        self.split_spectrogram_duration_s = split_spectrogram_duration_s
        self.slide_frames = slide_frames

    def get_random_spectrogram(self):
        """Retrieves a random audio clip's spectrogram that is optionally augmented.

        Returns:
            numpy.ndarry: 2D spectrogram array for the random (augmented) audio clip.
        """
        clip = self.clips.get_random_clip()
        if self.augmenter is not None:
            clip = self.augmenter.augment_clip(clip)

        return generate_features_for_clip(clip, self.step_ms)

    def spectrogram_generator(self, random=False, max_clips=None, **kwargs):
        """A Python generator that retrieves (augmented) spectrograms.

        Args:
            random (bool, optional): Specifies if the source audio clips should be chosen randomly. Defaults to False.
            kwargs: Parameters to pass to the clips audio generator.

        Yields:
            numpy.ndarry: 2D spectrogram array for the random (augmented) audio clip.
        """
        if random:
            if max_clips is not None:
                clip_generator = self.clips.random_audio_generator(max_clips=max_clips)
            else:
                clip_generator = self.clips.random_audio_generator()
        else:
            clip_generator = self.clips.audio_generator(**kwargs)

        if self.augmenter is not None:
            augmented_generator = self.augmenter.augment_generator(clip_generator)
        else:
            augmented_generator = clip_generator

        for augmented_clip in augmented_generator:
            spectrogram = generate_features_for_clip(augmented_clip, self.step_ms)

            if self.split_spectrogram_duration_s is not None:
                # Splits the resulting spectrogram into non-overlapping spectrograms. The features from the first 20 feature windows are dropped.
                desired_spectrogram_length = int(
                    self.split_spectrogram_duration_s / (self.step_ms / 1000)
                )

                if spectrogram.shape[0] > desired_spectrogram_length + 20:
                    slided_spectrograms = np.lib.stride_tricks.sliding_window_view(
                        spectrogram,
                        window_shape=(desired_spectrogram_length, spectrogram.shape[1]),
                    )[20::desired_spectrogram_length, ...]

                    for i in range(slided_spectrograms.shape[0]):
                        yield np.squeeze(slided_spectrograms[i])
                else:
                    yield spectrogram
            elif self.slide_frames is not None:
                # Generates self.slide_frames spectrograms by shifting over the already generated spectrogram
                spectrogram_length = spectrogram.shape[0] - self.slide_frames + 1

                slided_spectrograms = np.lib.stride_tricks.sliding_window_view(
                    spectrogram, window_shape=(spectrogram_length, spectrogram.shape[1])
                )
                for i in range(self.slide_frames):
                    yield np.squeeze(slided_spectrograms[i])
            else:
                yield spectrogram
```

```average_pooling2d.py
# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Convolutional AveragePooling2D."""
import numpy as np
import tensorflow as tf


class AveragePooling2D(tf.keras.layers.Layer):
    """AveragePooling2D layer.

    It is convolutional AveragePooling2D based on depthwise_conv2d.
    It can be useful for cases where AveragePooling2D has to run in streaming mode

    The input data with shape [batch_size, time1, feature1, feature2]
    are processed by depthwise conv with fixed weights, all weights values
    are equal to 1.0/(size_in_time_1*size_in_feature1).
    Averaging is done in 'time1' and 'feature1' dims.
    Conv filter has size [size_in_time_1, size_in_feature1, feature2],
    where first two dims are specified by user and
    feature2 is defiend by the last dim of input data.

    So if kernel_size = [time1, feature1]
    output will be [batch_size, time1, 1, feature2]

    Attributes:
      kernel_size: 2D kernel size - defines the dims
        which will be eliminated/averaged.
      strides: stride for each dim, with size 4
      padding: defiens how to pad
      dilation_rate: dilation rate in which we sample input values
        across the height and width
      **kwargs: additional layer arguments
    """

    def __init__(
        self, kernel_size, strides=None, padding="valid", dilation_rate=None, **kwargs
    ):
        super(AveragePooling2D, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        if not self.strides:
            self.strides = [1, 1, 1, 1]

        if not self.dilation_rate:
            self.dilation_rate = [1, 1]

    def build(self, input_shape):
        super(AveragePooling2D, self).build(input_shape)
        # expand filters shape with the last dimension
        filter_shape = self.kernel_size + (input_shape[-1],)
        self.filters = self.add_weight("kernel", shape=filter_shape)

        init_weight = np.ones(filter_shape) / np.prod(self.kernel_size)
        self.set_weights([init_weight])

    def call(self, inputs):
        # inputs [batch_size, time1, feature1, feature2]
        time_kernel_exp = tf.expand_dims(self.filters, -1)
        # it can be replaced by AveragePooling2D with temporal padding
        # and optimized for streaming mode
        # output will be [batch_size, time1, feature1, feature2]
        return tf.nn.depthwise_conv2d(
            inputs,
            time_kernel_exp,
            strides=self.strides,
            padding=self.padding.upper(),
            dilations=self.dilation_rate,
            name=self.name + "_averPool2D",
        )

    def get_config(self):
        config = super(AveragePooling2D, self).get_config()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.dilation_rate,
            }
        )
        return config
```

```delay.py
# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Dealy layer."""

from speech_commands.layers import modes
import tensorflow as tf


class Delay(tf.keras.layers.Layer):
    """Delay layer.

    It is useful for introducing delay in streaming mode for non causal filters.
    For example in residual connections with multiple conv layers

    Attributes:
      mode: Training or inference modes: non streaming, streaming.
      delay: delay value
      inference_batch_size: batch size in inference mode
      also_in_non_streaming: Apply delay also in training and non-streaming
        inference mode.
      **kwargs: additional layer arguments
    """

    def __init__(
        self,
        mode=modes.Modes.TRAINING,
        delay=0,
        inference_batch_size=1,
        also_in_non_streaming=False,
        **kwargs,
    ):
        super(Delay, self).__init__(**kwargs)
        self.mode = mode
        self.delay = delay
        self.inference_batch_size = inference_batch_size
        self.also_in_non_streaming = also_in_non_streaming

        if delay < 0:
            raise ValueError("delay (%d) must be non-negative" % delay)

    def build(self, input_shape):
        super(Delay, self).build(input_shape)

        if self.delay > 0:
            self.state_shape = [
                self.inference_batch_size,
                self.delay,
            ] + input_shape.as_list()[2:]
            if self.mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
                self.states = self.add_weight(
                    name="states",
                    shape=self.state_shape,
                    trainable=False,
                    initializer=tf.zeros_initializer,
                )

            elif self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
                # For streaming inference with extrnal states,
                # the states are passed in as input.
                self.input_state = tf.keras.layers.Input(
                    shape=self.state_shape[1:],
                    batch_size=self.inference_batch_size,
                    name=self.name + "/input_state_delay",
                )
                self.output_state = None

    def call(self, inputs):
        if self.delay == 0:
            return inputs

        if self.mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
            return self._streaming_internal_state(inputs)

        elif self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
            # in streaming inference mode with external state
            # in addition to the output we return the output state.
            output, self.output_state = self._streaming_external_state(
                inputs, self.input_state
            )
            return output

        elif self.mode in (modes.Modes.TRAINING, modes.Modes.NON_STREAM_INFERENCE):
            # run non streamable training or non streamable inference
            return self._non_streaming(inputs)

        else:
            raise ValueError(f"Encountered unexpected mode `{self.mode}`.")

    def get_config(self):
        config = super(Delay, self).get_config()
        config.update(
            {
                "mode": self.mode,
                "delay": self.delay,
                "inference_batch_size": self.inference_batch_size,
                "also_in_non_streaming": self.also_in_non_streaming,
            }
        )
        return config

    def _streaming_internal_state(self, inputs):
        memory = tf.keras.layers.concatenate([self.states, inputs], 1)
        outputs = memory[:, : inputs.shape.as_list()[1]]
        new_memory = memory[:, -self.delay :]
        assign_states = self.states.assign(new_memory)

        with tf.control_dependencies([assign_states]):
            return tf.identity(outputs)

    def _streaming_external_state(self, inputs, states):
        memory = tf.keras.layers.concatenate([states, inputs], 1)
        outputs = memory[:, : inputs.shape.as_list()[1]]
        new_memory = memory[:, -self.delay :]
        return outputs, new_memory

    def _non_streaming(self, inputs):
        if self.also_in_non_streaming:
            return tf.pad(
                inputs, ((0, 0), (self.delay, 0)) + ((0, 0),) * (inputs.shape.rank - 2)
            )[:, : -self.delay]
        else:
            return inputs

    def get_input_state(self):
        # input state will be used only for STREAM_EXTERNAL_STATE_INFERENCE mode
        if self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
            return [self.input_state]
        else:
            raise ValueError(
                "Expected the layer to be in external streaming mode, "
                f"not `{self.mode}`."
            )

    def get_output_state(self):
        # output state will be used only for STREAM_EXTERNAL_STATE_INFERENCE mode
        if self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
            return [self.output_state]
        else:
            raise ValueError(
                "Expected the layer to be in external streaming mode, "
                f"not `{self.mode}`."
            )
```

```modes.py
# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Modes the model can be in and its input data shape."""


class Modes(object):
    """Definition of the mode the model is functioning in."""

    # Model is in a training state. No streaming is done.
    TRAINING = "TRAINING"

    # Below are three options for inference:

    # Model is in inference mode and has state for efficient
    # computation/streaming, where state is kept inside of the model
    STREAM_INTERNAL_STATE_INFERENCE = "STREAM_INTERNAL_STATE_INFERENCE"

    # Model is in inference mode and has state for efficient
    # computation/streaming, where state is received from outside of the model
    STREAM_EXTERNAL_STATE_INFERENCE = "STREAM_EXTERNAL_STATE_INFERENCE"

    # Model its in inference mode and it's topology is the same with training
    # mode (with removed droputs etc)
    NON_STREAM_INFERENCE = "NON_STREAM_INFERENCE"


def get_input_data_shape(config, mode):
    """Gets data shape for a neural net input layer.

    Args:
      config: dictionary containing training parameters
      mode: inference mode described above at Modes

    Returns:
      data_shape for input layer
    """

    if mode not in (
        Modes.TRAINING,
        Modes.NON_STREAM_INFERENCE,
        Modes.STREAM_INTERNAL_STATE_INFERENCE,
        Modes.STREAM_EXTERNAL_STATE_INFERENCE,
    ):
        raise ValueError('Unknown mode "%s" ' % config["mode"])

    if mode in (Modes.TRAINING, Modes.NON_STREAM_INFERENCE):
        data_shape = (config["spectrogram_length"], 40)
    else:
        stride = config['stride']
        data_shape = (stride, 40)
    return data_shape
```

```strided_drop.py
# coding=utf-8
# Copyright 2024 Kevin Ahrendt.
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

import tensorflow as tf

from speech_commands.layers import modes


class StridedDrop(tf.keras.layers.Layer):
    """StridedDrop

    Drops the specified audio feature slices in nonstreaming mode only.
    Used for matching the dimensions of convolutions with valid padding.

    Attributes:
        time_sclices_to_drop: number of audio feature slices to drop
        mode: inference mode; e.g., non-streaming, internal streaming
    """

    def __init__(
        self, time_slices_to_drop, mode=modes.Modes.NON_STREAM_INFERENCE, **kwargs
    ):
        super(StridedDrop, self).__init__(**kwargs)
        self.time_slices_to_drop = time_slices_to_drop
        self.mode = mode
        self.state_shape = []

    def call(self, inputs):
        if self.mode == modes.Modes.NON_STREAM_INFERENCE:
            inputs = inputs[:, self.time_slices_to_drop :, :, :]
            return inputs

        return inputs

    def get_config(self):
        config = {
            "time_slices_to_drop": self.time_slices_to_drop,
            "mode": self.mode,
        }
        base_config = super(StridedDrop, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_input_state(self):
        return []

    def get_output_state(self):
        return []


class StridedKeep(tf.keras.layers.Layer):
    """StridedKeep

    Keeps the specified audio feature slices in streaming mode only.
    Used for splitting a single streaming ring buffer into multiple branches with minimal overhead.

    Attributes:
        time_sclices_to_keep: number of audio feature slices to keep
        mode: inference mode; e.g., non-streaming, internal streaming
    """

    def __init__(
        self, time_slices_to_keep, mode=modes.Modes.NON_STREAM_INFERENCE, **kwargs
    ):
        super(StridedKeep, self).__init__(**kwargs)
        self.time_slices_to_keep = max(time_slices_to_keep, 1)
        self.mode = mode
        self.state_shape = []

    def call(self, inputs):
        if self.mode != modes.Modes.NON_STREAM_INFERENCE:
            return inputs[:, -self.time_slices_to_keep :, :, :]

        return inputs

    def get_config(self):
        config = {
            "time_slices_to_keep": self.time_slices_to_keep,
            "mode": self.mode,
        }
        base_config = super(StridedKeep, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_input_state(self):
        return []

    def get_output_state(self):
        return []
```

```sub_spectral_normalization.py
# coding=utf-8
# Copyright 2023 The Google Research Authors.
# Modifications copyright 2024 Kevin Ahrendt.
#
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

"""Sub spectral normalization layer."""
from typing import Any, Dict

import tensorflow as tf


class SubSpectralNormalization(tf.keras.layers.Layer):
    """Sub spectral normalization layer.

    It is based on paper:
    "SUBSPECTRAL NORMALIZATION FOR NEURAL AUDIO DATA PROCESSING"
    https://arxiv.org/pdf/2103.13620.pdf
    """

    def __init__(self, sub_groups, **kwargs):
        super(SubSpectralNormalization, self).__init__(**kwargs)
        self.sub_groups = sub_groups

        self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        # expected input: [N, Time, Frequency, Channels]
        if inputs.shape.rank != 4:
            raise ValueError("input_shape.rank:%d must be 4" % inputs.shape.rank)

        input_shape = inputs.shape.as_list()
        if input_shape[3] % self.sub_groups:
            raise ValueError(
                "input_shape[3]: %d must be divisible by "
                "self.sub_groups %d " % (input_shape[3], self.sub_groups)
            )

        net = inputs
        if self.sub_groups == 1:
            net = self.batch_norm(net)
        else:
            target_shape = [
                input_shape[1],
                input_shape[3] // self.sub_groups,
                input_shape[2] * self.sub_groups,
            ]
            net = tf.keras.layers.Reshape(target_shape)(net)
            net = self.batch_norm(net)
            net = tf.keras.layers.Reshape(input_shape[1:])(net)
        return net

    def get_config(self):
        config = {"sub_groups": self.sub_groups}
        base_config = super(SubSpectralNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
```

```training_parameters.yaml
batch_size: 128
clip_duration_ms: 1500
epochs: 300
eval_step_interval: 500
features:
- features_dir: commands_augmented/negative/speech
  label: 0
  penalty_weight: 1.0
  sampling_weight: 10.0
  truncation_strategy: random
  truth: false
  type: mmap
- features_dir: commands_augmented/negative/dinner_party
  label: 0
  penalty_weight: 1.0
  sampling_weight: 10.0
  truncation_strategy: random
  truth: false
  type: mmap
- features_dir: commands_augmented/negative/no_speech
  label: 0
  penalty_weight: 1.0
  sampling_weight: 5.0
  truncation_strategy: random
  truth: false
  type: mmap
- features_dir: commands_augmented/negative/dinner_party_eval
  label: 0
  penalty_weight: 1.0
  sampling_weight: 1.0
  truncation_strategy: split
  truth: false
  type: mmap
- features_dir: commands_augmented/commands/adelante
  label: 1
  penalty_weight: 1.0
  sampling_weight: 1.0
  truncation_strategy: truncate_start
  truth: true
  type: mmap
- features_dir: commands_augmented/commands/atras
  label: 2
  penalty_weight: 1.0
  sampling_weight: 1.0
  truncation_strategy: truncate_start
  truth: true
  type: mmap
- features_dir: commands_augmented/commands/derecha
  label: 3
  penalty_weight: 1.0
  sampling_weight: 1.0
  truncation_strategy: truncate_start
  truth: true
  type: mmap
- features_dir: commands_augmented/commands/frena
  label: 4
  penalty_weight: 1.0
  sampling_weight: 1.0
  truncation_strategy: truncate_start
  truth: true
  type: mmap
- features_dir: commands_augmented/commands/izquierda
  label: 5
  penalty_weight: 1.0
  sampling_weight: 1.0
  truncation_strategy: truncate_start
  truth: true
  type: mmap
- features_dir: commands_augmented/commands/lento
  label: 6
  penalty_weight: 1.0
  sampling_weight: 1.0
  truncation_strategy: truncate_start
  truth: true
  type: mmap
- features_dir: commands_augmented/commands/moderado
  label: 7
  penalty_weight: 1.0
  sampling_weight: 1.0
  truncation_strategy: truncate_start
  truth: true
  type: mmap
- features_dir: commands_augmented/commands/rapido
  label: 8
  penalty_weight: 1.0
  sampling_weight: 1.0
  truncation_strategy: truncate_start
  truth: true
  type: mmap
- features_dir: commands_augmented/commands/reversa
  label: 9
  penalty_weight: 1.0
  sampling_weight: 1.0
  truncation_strategy: truncate_start
  truth: true
  type: mmap
freq_mask_count:
- 0
freq_mask_max_size:
- 0
labels:
  adelante: 1
  atras: 2
  derecha: 3
  frena: 4
  izquierda: 5
  lento: 6
  moderado: 7
  negative: 0
  rapido: 8
  reversa: 9
learning_rates:
- 0.001
maximization_metric: average_viable_recall
minimization_metric: null
negative_class_weight:
- 20
num_classes: 10
positive_class_weight:
- 1
target_minimization: 0.9
time_mask_count:
- 0
time_mask_max_size:
- 0
train_dir: trained
training_steps:
- 20000
window_step_ms: 10
```

```main.py
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

import argparse
import os
import sys
import yaml
import platform
from absl import logging

import tensorflow as tf

# Disable GPU by default on ARM Macs, it's slower than just using the CPU
if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" or (
    sys.platform == "darwin"
    and platform.processor() == "arm"
    and "CUDA_VISIBLE_DEVICES" not in os.environ
):
    tf.config.set_visible_devices([], "GPU")

import speech_commands.data as input_data
import speech_commands.train as train
import speech_commands.test as test
import speech_commands.utils as utils

import speech_commands.mixednet as mixednet

from speech_commands.layers import modes


def load_config(flags, model_module):
    """Loads the training configuration from the specified yaml file.

    Args:
        flags (argparse.Namespace): command line flags
        model_module (module): python module for loading the model

    Returns:
        dict: dictionary containing training configuration
    """
    config_filename = flags.training_config
    config = yaml.load(open(config_filename, "r").read(), yaml.Loader)

    config["summaries_dir"] = os.path.join(config["train_dir"], "logs/")

    config["stride"] = flags.__dict__.get("stride", 1)
    config["window_step_ms"] = config.get("window_step_ms", 20)

    config["num_classes"] = config.get("num_classes", 2)
    config["commands_dir"] = config.get("commands_dir", "speech-commands-spanish/commands")


    # Default preprocessor settings
    preprocessor_sample_rate = 16000  # Hz
    preprocessor_window_size = 30  # ms
    preprocessor_window_step = config["window_step_ms"]  # ms

    desired_samples = int(preprocessor_sample_rate * config["clip_duration_ms"] / 1000)

    window_size_samples = int(
        preprocessor_sample_rate * preprocessor_window_size / 1000
    )
    window_step_samples = int(
        config["stride"] * preprocessor_sample_rate * preprocessor_window_step / 1000
    )

    length_minus_window = desired_samples - window_size_samples

    if length_minus_window < 0:
        config["spectrogram_length_final_layer"] = 0
    else:
        config["spectrogram_length_final_layer"] = 1 + int(
            length_minus_window / window_step_samples
        )

    config["spectrogram_length"] = config[
        "spectrogram_length_final_layer"
    ] + model_module.spectrogram_slices_dropped(flags)

    config["flags"] = flags.__dict__

    config["training_input_shape"] = modes.get_input_data_shape(
        config, modes.Modes.TRAINING
    )

    return config


def train_model(config, model, data_processor, restore_checkpoint):
    """Trains a model.

    Args:
        config (dict): dictionary containing training configuration
        model (Keras model): model architecture to train
        data_processor (FeatureHandler): feature handler that loads spectrogram data
        restore_checkpoint (bool): Whether to restore from checkpoint if model exists

    Raises:
        ValueError: If the model exists but the training flag isn't set
    """
    try:
        os.makedirs(config["train_dir"])
        os.mkdir(config["summaries_dir"])
    except OSError:
        if restore_checkpoint:
            pass
        else:
            raise ValueError(
                "model already exists in folder %s" % config["train_dir"]
            ) from None
    config_fname = os.path.join(config["train_dir"], "training_config.yaml")

    with open(config_fname, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    utils.save_model_summary(model, config["train_dir"])

    train.train(model, config, data_processor)


def evaluate_model(
    config,
    model,
    data_processor,
    test_tf_nonstreaming,
    test_tflite_nonstreaming,
    test_tflite_nonstreaming_quantized,
    test_tflite_streaming,
    test_tflite_streaming_quantized,
):
    """Evaluates a model on test data.

    Saves the nonstreaming model or streaming model in SavedModel format,
    then converts it to TFLite as specified.

    Args:
        config (dict): dictionary containing training configuration
        model (Keras model): model (with loaded weights) to test
        data_processor (FeatureHandler): feature handler that loads spectrogram data
        test_tf_nonstreaming (bool): Evaluate the nonstreaming SavedModel
        test_tflite_nonstreaming_quantized (bool): Convert and evaluate quantized nonstreaming TFLite model
        test_tflite_nonstreaming (bool): Convert and evaluate nonstreaming TFLite model
        test_tflite_streaming (bool): Convert and evaluate streaming TFLite model
        test_tflite_streaming_quantized (bool): Convert and evaluate quantized streaming TFLite model
    """

    if (
        test_tf_nonstreaming
        or test_tflite_nonstreaming
        or test_tflite_nonstreaming_quantized
    ):
        # Save the nonstreaming model to disk
        logging.info("Saving nonstreaming model")

        utils.convert_model_saved(
            model,
            config,
            folder="non_stream",
            mode=modes.Modes.NON_STREAM_INFERENCE,
        )

    if test_tflite_streaming or test_tflite_streaming_quantized:
        # Save the internal streaming model to disk
        logging.info("Saving streaming model")

        utils.convert_model_saved(
            model,
            config,
            folder="stream_state_internal",
            mode=modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
        )

    if test_tf_nonstreaming:
        logging.info("Testing nonstreaming model")

        folder_name = "non_stream"
        test.tf_model_accuracy(
            config,
            folder_name,
            data_processor,
            data_set="testing",
            accuracy_name="testing_set_metrics.txt",
        )

    tflite_configs = []

    if test_tflite_nonstreaming:
        tflite_configs.append(
            {
                "log_string": "nonstreaming model",
                "source_folder": "non_stream",
                "output_folder": "tflite_non_stream",
                "filename": "non_stream.tflite",
                "testing_dataset": "testing",
                "testing_ambient_dataset": "testing_ambient",
                "quantize": False,
            }
        )

    if test_tflite_nonstreaming_quantized:
        tflite_configs.append(
            {
                "log_string": "quantized nonstreaming model",
                "source_folder": "non_stream",
                "output_folder": "tflite_non_stream_quant",
                "filename": "non_stream_quant.tflite",
                "testing_dataset": "testing",
                "testing_ambient_dataset": "testing_ambient",
                "quantize": True,
            }
        )

    for tflite_config in tflite_configs:
        logging.info("Converting %s to TFLite", tflite_config["log_string"])

        utils.convert_saved_model_to_tflite(
            config,
            audio_processor=data_processor,
            path_to_model=os.path.join(config["train_dir"], tflite_config["source_folder"]),
            folder=os.path.join(config["train_dir"], tflite_config["output_folder"]),
            fname=tflite_config["filename"],
            quantize=tflite_config["quantize"],
        )

        logging.info(
            "Testing the TFLite %s false accept per hour and false rejection rates at various cutoffs.",
            tflite_config["log_string"],
        )

        test.tflite_model_accuracy(
            config,
            tflite_config["output_folder"],
            data_processor,
            data_set=tflite_config["testing_dataset"],
            # ambient_set=tflite_config["testing_ambient_dataset"],
            tflite_model_name=tflite_config["filename"],
            accuracy_name="tflite_model_accuracy.txt",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_config",
        type=str,
        default="trained_models/model/training_parameters.yaml",
        help="""\
        Path to the training parameters yaml configuration.action=
        """,
    )
    parser.add_argument(
        "--train",
        type=int,
        default=1,
        help="If 1 run train and test, else run only test",
    )
    parser.add_argument(
        "--test_tf_nonstreaming",
        type=int,
        default=0,
        help="Save the nonstreaming model and test on the test datasets",
    )
    parser.add_argument(
        "--test_tflite_nonstreaming",
        type=int,
        default=0,
        help="Save the TFLite nonstreaming model and test on the test datasets",
    )
    parser.add_argument(
        "--test_tflite_nonstreaming_quantized",
        type=int,
        default=0,
        help="Save the TFLite quantized nonstreaming model and test on the test datasets",
    )
    parser.add_argument(
        "--test_tflite_streaming",
        type=int,
        default=0,
        help="Save the (non-quantized) streaming model and test on the test datasets",
    )
    parser.add_argument(
        "--test_tflite_streaming_quantized",
        type=int,
        default=1,
        help="Save the quantized streaming model and test on the test datasets",
    )
    parser.add_argument(
        "--restore_checkpoint",
        type=int,
        default=0,
        help="If 1 it will restore a checkpoint and resume the training "
        "by initializing model weights and optimizer with checkpoint values. "
        "It will use learning rate and number of training iterations from "
        "--learning_rate and --how_many_training_steps accordinlgy. "
        "This option is useful in cases when training was interrupted. "
        "With it you should adjust learning_rate and how_many_training_steps.",
    )
    parser.add_argument(
        "--use_weights",
        type=str,
        default="best_weights",
        help="Which set of weights to use when creating the model"
        "One of `best_weights`` or `last_weights`.",
    )

    # Function used to parse --verbosity argument
    def verbosity_arg(value):
        """Parses verbosity argument.

        Args:
        value: A member of tf.logging.

        Returns:
        TF logging mode

        Raises:
        ArgumentTypeError: Not an expected value.
        """
        value = value.upper()
        if value == "INFO":
            return logging.INFO
        elif value == "DEBUG":
            return logging.DEBUG
        elif value == "ERROR":
            return logging.ERROR
        elif value == "FATAL":
            return logging.FATAL
        elif value == "WARN":
            return logging.WARN
        else:
            raise argparse.ArgumentTypeError("Not an expected value")

    parser.add_argument(
        "--verbosity",
        type=verbosity_arg,
        default=logging.INFO,
        help='Log verbosity. Can be "INFO", "DEBUG", "ERROR", "FATAL", or "WARN"',
    )

    # sub parser for model settings
    subparsers = parser.add_subparsers(dest="model_name", help="NN model name")

    # mixednet model settings
    parser_mixednet = subparsers.add_parser("mixednet")
    mixednet.model_parameters(parser_mixednet)

    flags, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError("Unknown argument: {}".format(unparsed))

    if flags.model_name == "mixednet":
        model_module = mixednet
    else:
        raise ValueError("Unknown model type: {}".format(flags.model_name))

    logging.set_verbosity(flags.verbosity)

    config = load_config(flags, model_module)

    data_processor = input_data.FeatureHandler(config)

    if flags.train:
        model = model_module.model(
            flags, config["training_input_shape"], config["batch_size"], config["num_classes"]
        )
        model.summary(line_length=100)
        train_model(config, model, data_processor, flags.restore_checkpoint)
    else:
        if not os.path.isdir(config["train_dir"]):
            raise ValueError('model is not trained set "--train 1" and retrain it')

    if (
        flags.test_tf_nonstreaming
        or flags.test_tflite_nonstreaming
        or flags.test_tflite_streaming
        or flags.test_tflite_streaming_quantized
    ):
        model = model_module.model(
            flags, shape=config["training_input_shape"], batch_size=1, num_classes=config["num_classes"]
        )

        model.load_weights(
            os.path.join(config["train_dir"], flags.use_weights) + ".weights.h5"
        )

        logging.info(model.summary())

        evaluate_model(
            config,
            model,
            data_processor,
            flags.test_tf_nonstreaming,
            flags.test_tflite_nonstreaming,
            flags.test_tflite_nonstreaming_quantized,
            flags.test_tflite_streaming,
            flags.test_tflite_streaming_quantized,
        )
```

