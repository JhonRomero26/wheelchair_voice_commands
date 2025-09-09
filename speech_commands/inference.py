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
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()

        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()

        # Metadata de entrada/salida
        in_shape = self.input_details[0]["shape"]  # p.ej., [1, T, F] o [1, T, F, 1]
        if len(in_shape) == 3:
            # [B, T, F]
            self.input_feature_slices = int(in_shape[1])   # T
            self.input_feature_bins   = int(in_shape[2])   # F
        elif len(in_shape) == 4:
            # [B, T, F, C]
            self.input_feature_slices = int(in_shape[1])   # T
            self.input_feature_bins   = int(in_shape[2])   # F
        else:
            raise ValueError(f"Forma de entrada no soportada: {in_shape}")

        out_shape = self.output_details[0]["shape"]
        self.num_classes = int(out_shape[-1])

        self.is_quantized_model = self.input_details[0]["dtype"] in (np.int8, np.uint8)

        self.stride = self.input_feature_slices if stride is None else int(stride)

        # Primear tensores con ceros
        for s in range(len(self.input_details)):
            zero = np.zeros(self.input_details[s]["shape"], dtype=self.input_details[s]["dtype"])
            interpreter.set_tensor(self.input_details[s]["index"], zero)

        self.model = interpreter

    # ---------- Helpers de cuantización (INT8/UINT8) ----------
    def _get_qparams(self, details: dict):
        """Obtiene (scale, zero_point) tolerando variantes de clave entre TF / LiteRT."""
        qp = details.get("quantization_parameters") or {}
        scales = qp.get("scales") or []
        zeros  = qp.get("zero_points") or []
        if len(scales) and len(zeros):
            return float(scales[0]), int(zeros[0])
        # Fallback para claves antiguas
        q = details.get("quantization")  # (scale, zero_point) o (0.0, 0)
        if isinstance(q, (tuple, list)) and len(q) == 2:
            return float(q[0] or 1.0), int(q[1] or 0)
        # Sin cuantización declarada
        return 1.0, 0

    def quantize_input_data(self, data: np.ndarray, input_details: dict) -> np.ndarray:
        """Cuantiza data float32 -> (int8/uint8) usando scale/zero_point del tensor de entrada."""
        dtype = input_details["dtype"]
        scale, zero = self._get_qparams(input_details)
        if scale == 0:
            scale = 1.0
        q = np.round(data / scale + zero)
        # Clip al rango del dtype
        if dtype == np.int8:
            q = np.clip(q, -128, 127).astype(np.int8, copy=False)
        elif dtype == np.uint8:
            q = np.clip(q, 0, 255).astype(np.uint8, copy=False)
        else:
            # Si el modelo no es int8/uint8, devuelve float32 tal cual
            return data.astype(np.float32, copy=False)
        return q

    def dequantize_output_data(self, data: np.ndarray, output_details: dict) -> np.ndarray:
        """Descuantiza salida (int8/uint8) -> float32 usando scale/zero_point del tensor de salida."""
        scale, zero = self._get_qparams(output_details)
        if scale == 0:
            scale = 1.0
        return (data.astype(np.float32) - float(zero)) * float(scale)

    def _to_int16_pcm(self, data: np.ndarray) -> np.ndarray:
        """Asegura int16 PCM a 16 kHz (asumiendo ya muestreado a 16 kHz)."""
        if data.dtype == np.int16:
            return data
        if np.issubdtype(data.dtype, np.floating):
            x = np.clip(data, -1.0, 1.0)
            return (x * 32767.0).astype(np.int16)
        # otros dtypes → cast directo
        return data.astype(np.int16)


    def _pad_spectrogram_frames(self, spec: np.ndarray) -> np.ndarray:
        """Si spec tiene menos frames de los que el modelo espera (T), pad con ceros arriba."""
        t_needed = self.input_feature_slices
        t_have   = int(spec.shape[0])
        if t_have >= t_needed:
            return spec
        pad_rows = t_needed - t_have
        pad = np.zeros((pad_rows, spec.shape[1]), dtype=spec.dtype)
        return np.vstack([pad, spec])


    def predict_clip(self, data: np.ndarray, step_ms: int = 20):
        """
        Ejecuta el modelo sobre un clip crudo (mono, 16 kHz).
        Retorna np.ndarray de forma (num_chunks, num_classes) o (1, num_classes).
        """
        # Normaliza a int16 PCM si viene en float32
        data = self._to_int16_pcm(data)

        # Extrae espectrograma (usa el mismo step_ms que en entrenamiento)
        spectrogram = generate_features_for_clip(data, step_ms=step_ms)

        return self.predict_spectrogram(spectrogram)


    def predict_spectrogram(self, spectrogram: np.ndarray):
        """
        Ejecuta el modelo sobre un espectrograma 2D [T, F] (T=time-slices, F=feature-bins).
        Hace padding si T < T_expected para garantizar al menos 1 chunk.
        Retorna np.ndarray de forma (num_chunks, num_classes).
        """
        # Normaliza dtype de features
        if np.issubdtype(spectrogram.dtype, np.uint16):
            # Escala típica de microfrontend (ajuste suave)
            spectrogram = spectrogram.astype(np.float32) * 0.0390625
        elif np.issubdtype(spectrogram.dtype, np.float64):
            spectrogram = spectrogram.astype(np.float32)
        else:
            spectrogram = spectrogram.astype(np.float32, copy=False)

        # Si hay muy pocas filas, pad hasta T esperado
        if spectrogram.shape[0] < self.input_feature_slices:
            spectrogram = self._pad_spectrogram_frames(spectrogram)

        # Slicing en ventanas de T (stride por defecto = T => una sola ventana)
        chunks = []
        for last_index in range(self.input_feature_slices, len(spectrogram) + 1, self.stride):
            chunk = spectrogram[last_index - self.input_feature_slices : last_index]  # [T, F]
            if chunk.shape[0] == self.input_feature_slices:
                chunks.append(chunk)

        if not chunks:
            # Asegura un “batch” vacío seguro con la forma correcta
            return np.zeros((0, self.num_classes), dtype=np.float32)

        predictions = []
        for chunk in chunks:
            x = chunk.astype(np.float32, copy=False)
            if self.is_quantized_model and x.dtype not in (np.int8, np.uint8):
                x = self.quantize_input_data(x, self.input_details[0])

            x = np.reshape(x, self.input_details[0]["shape"])
            self.model.set_tensor(self.input_details[0]["index"], x)
            self.model.invoke()

            raw = self.model.get_tensor(self.output_details[0]["index"])[0]

            if self.output_details[0]["dtype"] in (np.uint8, np.int8):
                raw = self.dequantize_output_data(raw, self.output_details[0])

            predictions.append(raw.astype(np.float32, copy=False))

        return np.stack(predictions, axis=0)  # (num_chunks, num_classes)