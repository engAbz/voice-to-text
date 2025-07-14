from flask import Flask, request, jsonify
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import os
from functools import lru_cache
from queue import Queue
from threading import Thread
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
request_queue = Queue()

@lru_cache(maxsize=1)
def load_model():
    logger.info("Loading Whisper model...")
    processor = WhisperProcessor.from_pretrained("ayoubkirouane/whisper-small-ar")
    model = WhisperForConditionalGeneration.from_pretrained("ayoubkirouane/whisper-small-ar")
    return processor, model

processor, model = load_model()

def process_audio(audio, sr):
    logger.info("Processing audio...")
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features, language="ar")
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    logger.info("Transcription completed.")
    return transcription

def worker():
    while True:
        audio, sr, callback = request_queue.get()
        try:
            transcription = process_audio(audio, sr)
            callback(transcription)
        except Exception as e:
            logger.error(f"Error in worker: {e}")
            callback(f"Error: {str(e)}")
        request_queue.task_done()

Thread(target=worker, daemon=True).start()

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        if "audio" not in request.files:
            logger.error("No audio file provided")
            return jsonify({"error": "No audio file provided"}), 400
        audio_file = request.files["audio"]
        audio_path = "temp_audio.mp3"
        audio_file.save(audio_path)
        audio, sr = librosa.load(audio_path, sr=16000)
        result = []
        def callback(transcription):
            result.append(transcription)
        request_queue.put((audio, sr, callback))
        request_queue.join()
        os.remove(audio_path)
        logger.info(f"Transcription: {result[0]}")
        return jsonify({"transcription": result[0]})
    except Exception as e:
        logger.error(f"Error in transcribe: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
