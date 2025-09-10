# realtime_min.py
import numpy as np
import sounddevice as sd
import yaml

from speech_commands.inference import Model  # tu wrapper con predict_clip

# === Configuración básica ===
MODEL_PATH   = "./trained/tflite_non_stream_quant/non_stream_quant.tflite"
SAMPLE_RATE  = 16000           # Hz
WINDOW_SEC   = 3             # segundos por inferencia (ajústalo si tu modelo usa otra ventana)
BLOCK_SIZE   = 1024            # tamaño de bloque de captura
CHANNELS     = 1

model = Model(MODEL_PATH)

# Orden de clases (debe coincidir con el modelo)
CLASSES = [] 
with open("./training_parameters.yaml", "rt") as f:
    config = yaml.safe_load(f)
    labels = config["labels"]
    labels = dict(sorted(labels.items(), key=lambda item: item[1]))
    CLASSES = list(labels.keys())

print(CLASSES)

# === Estado mínimo ===
window_samples = int(round(WINDOW_SEC * SAMPLE_RATE))
audio_buffer = np.zeros(window_samples, dtype=np.float32)
audio_idx = 0

def audio_callback(indata, frames, time_info, status):
    global audio_buffer, audio_idx, model
    if status:
        print(status)

    mono = indata[:, 0].astype(np.float32, copy=False)
    audio_idx += len(mono)

    if mono.size >= audio_buffer.size:
        audio_buffer[:] = mono[-audio_buffer.size:]
    else:
        audio_buffer = np.concatenate([audio_buffer[mono.size:], mono], axis=0)
    
    if audio_idx >= window_samples:
        audio_idx = 0
        preds = model.predict_clip(audio_buffer)   # (N, C)
        if preds.shape[0] > 0:
            probs = preds[-1]                     # toma el último chunk
            idx = int(np.argmax(probs))
            print(CLASSES[idx], float(probs[idx]))



def main():
    # Carga modelo TFLite (no-stream)
    assert hasattr(model, "predict_clip"), "El wrapper Model debe exponer predict_clip(wave)."

    print("✅ Escuchando… (Ctrl+C para salir)")

    with sd.InputStream(channels=CHANNELS,
                        samplerate=SAMPLE_RATE,
                        blocksize=BLOCK_SIZE,
                        dtype="float32",
                        callback=audio_callback):
        while True:
            sd.sleep(100)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Detenido por el usuario.")
