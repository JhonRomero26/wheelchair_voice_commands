import numpy as np
import librosa.filters 
import sounddevice as sd
from ai_edge_litert.interpreter import Interpreter
from speech_commands.inference import Model
import matplotlib.pyplot as plt

# Cargar el modelo TFLite
# model_path = './trained/non_stream/saved_model.pb'
model_path = './alexa.tflite'
# model_path = './modelo_convertido.tflite'
model = Model(model_path)

# Configuración del audio
sample_rate = 16000  # Tasa de muestreo
block_size = 1024    # Tamaño del bloque
max_duration = 2   # Duración máxima del buffer de audio (segundos)
last_sample = sample_rate * max_duration


# Buffer para almacenar características
buffer_predictions = []

# Nombres de las clases
classes_names = [
    "silencio",
    "izquierda",
    "reversa",
    "atras",
    "lento",
    "frena",
    "derecha",
    "moderado",
    "rapido",
    "adelante"
]

# def audio_callback(indata, frames, time, status):
#     """Callback para procesar cada bloque de audio."""
#     global buffer_predictions

#     if status:
#         print(status)

#     buffer_predictions = np.append(buffer_predictions, indata)

#     if (len(buffer_predictions) > last_sample):
#         audio_clip = buffer_predictions[:int(last_sample)]
#         buffer_predictions = []

#         prediction = model.predict_clip(audio_clip)
#         # prediction = model.predict_clip(indata)
#         print(np.sum(prediction), np.sum(prediction) > 0.5)
#         # buffer_lenght = buffer_predictions.shape[0]

#     # if (buffer_lenght >  ):
#     # print(buffer_lenght)

def audio_callback(indata, frames, time, status):
    """Callback para procesar cada bloque de audio."""
    global buffer_predictions

    if status:
        print(status)

    prediction = model.predict_clip(indata)
    buffer_predictions = np.append(buffer_predictions, prediction)

    # if (len(buffer_predictions) > last_sample):
    #     audio_clip = buffer_predictions[:int(last_sample)]
    #     buffer_predictions = []

    #     prediction = model.predict_clip(audio_clip)
    #     print(np.sum(prediction), np.sum(prediction) > 0.5)

        

buffer_predictions
def start_stream():
    print("Escuchando audio... Presiona Ctrl+C para detener.")
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=block_size):
        while True:
            sd.sleep(100)


if __name__ == "__main__":
    try:
        start_stream()
    except KeyboardInterrupt:
        print("\nStreaming detenido.")