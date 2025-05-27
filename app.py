import os
import torch
import librosa
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from ast_models import ASTModel

app = Flask(__name__)
# Significantly increase Socket.IO timeout values
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    ping_timeout=300,  # 5 minutes timeout
    ping_interval=25,
    async_mode="threading"  # Use threading mode for better stability
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASTModel(label_dim=20, fstride=10, tstride=10, input_fdim=128, input_tdim=688, audioset_pretrain=False)
#label_dim: violin 20, cello 16 // input_tdim violin 688, cello 306
model.to(device)
model.load_state_dict(torch.load("ast_model_newGA.pth", map_location=device))
model.eval()

#violin
klasses = ['good-sound', 'crescendo', 'decrescendo', 'tremolo', 'vibrato', 'errors', 'bad-pitch', 'bad-dynamics',
           'bad-timbre', 'bad-richness', 'bad-attack', 'bad-rhythm', 'bridge', 'sultasto', 'pressure', 'rebound',
           'scale-good', 'staccato', 'minor', 'dirt']

# Preprocess audio chunk
def preprocess_audio(audio_chunk, sr=16000, max_time_steps=688):   
    #max_time_steps:688 violin, 306 cello 
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_chunk, sr=sr, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    if mel_spectrogram_db.shape[1] < max_time_steps:
        padding = max_time_steps - mel_spectrogram_db.shape[1]
        mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, padding)), mode='constant')
    else:
        mel_spectrogram_db = mel_spectrogram_db[:, :max_time_steps]

    return torch.tensor(mel_spectrogram_db, dtype=torch.float32).unsqueeze(0).to(device)

# Compute all predictions at once with improved chunk handling
def compute_predictions(audio_path, sr=16000):
    try:
        # Load the audio file
        y, _ = librosa.load(audio_path, sr=sr)
        
        # Calculate the duration of the audio in seconds
        audio_duration = librosa.get_duration(y=y, sr=sr)
        print(f"Audio duration: {audio_duration} seconds")
        
        # Define chunk size based on the audio duration - use larger chunks for longer files
        # if audio_duration <= 5:
        #     chunk_size = 1  # 1 second chunks
        # elif 5 < audio_duration <= 15:
        #     chunk_size = 2  # 2 seconds chunks
        # elif 15 < audio_duration <= 60:
        #     chunk_size = 3  # 3 seconds chunks
        # elif 60 < audio_duration <= 180:
        #     chunk_size = 5  # 5 seconds chunks
        # else:
        #     chunk_size = 10  # 10 seconds chunks for very long audio (3min+)

        chunk_size = 5

        print(f"Audio will be divided into chunks of {chunk_size} seconds")
        chunk_length = sr * chunk_size  # Length of each chunk in samples
        predictions_per_chunk = {}
        
        # Calculate total chunks more precisely to prevent index issues
        total_chunks = int(np.ceil(len(y) / chunk_length))
        
        # Process in batches to prevent socket timeouts
        batch_size = 20  # Process 20 chunks at a time
        for batch_start in range(0, total_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_chunks)
            
            # Process each chunk in the current batch
            for sec in range(batch_start, batch_end):
                start = sec * chunk_length
                end = min(start + chunk_length, len(y))
                
                # Skip if we've reached the end of the audio
                if start >= len(y):
                    break
                    
                audio_chunk = y[start:end]
                
                # Skip chunks that are too short
                if len(audio_chunk) < sr * 0.5:  # At least 0.5 seconds
                    continue

                mel_input = preprocess_audio(audio_chunk)
                with torch.no_grad():
                    outputs = model(mel_input)
                    predictions = torch.sigmoid(outputs).cpu().numpy()[0]

                threshold = 0.65    #the lower the more predicted
                predicted_labels = [klasses[i] for i, score in enumerate(predictions) if score > threshold]

                # Print predictions in console
                print(f"Segment {sec + 1} - Predicted Labels: {predicted_labels}")
                predictions_per_chunk[sec] = predicted_labels
            
            # Send a keepalive ping after each batch to keep socket connection alive
            socketio.emit("keep_alive", {"batch_processed": batch_end})

        # Notify client that processing is complete
        socketio.emit("processing_complete", {
            "predictions": predictions_per_chunk,
            "chunk_size": chunk_size,
            "filename": os.path.basename(audio_path)
        })

        return predictions_per_chunk, chunk_size
    
    except Exception as e:
        print(f"Error in processing audio: {str(e)}")
        socketio.emit("processing_error", {"error": str(e)})
        return {}, 0

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Start processing in the background - return immediately
    socketio.start_background_task(compute_predictions, file_path)
    
    return jsonify({
        "success": True, 
        "message": "Processing started",
        "filename": filename
    })

if __name__ == "__main__":
    socketio.run(app, debug=True)