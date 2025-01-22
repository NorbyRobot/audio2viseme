import librosa
import numpy as np
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from queue import Queue
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from datetime import timedelta
import re
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Global Constants
UPLOAD_FOLDER = 'audio_files'
PROCESSED_FOLDER = 'processed_files'
MODEL_PATH = 'model/audio2pho_model_mfa13label_ep300_1e-4_32.h5'  # Update with your model path
ALLOWED_EXTENSIONS = {'wav'}

VISEME_LABELS = {
    0: "Silence", 1: "P/B/M (Bilabial)", 2: "F/V (Labiodental)",
    3: "TH (Dental)", 4: "T/D/N/L (Alveolar)", 5: "S/Z (Sibilant)",
    6: "SH/CH/JH/ZH (Palato-alveolar)", 7: "K/G/NG (Velar)", 
    8: "H (Glottal)", 9: "R/W (Labio-velar)", 10: "Y (Palatal)",
    11: "EY/AE/AH (Vowel)", 12: "Rest Position"
}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def convert_time_to_offset(time_str):
    """Convert time string to millisecond offset"""
    hours, minutes, seconds = time_str.split(':')
    seconds, microseconds = seconds.split('.') if '.' in seconds else (seconds, '0')
    OFFSET = 0

    total_ms = (int(hours) * 3600000 +  # hours to ms
                int(minutes) * 60000 +   # minutes to ms
                int(seconds) * 1000 +    # seconds to ms
                int(float('0.' + microseconds) * 1000))  - OFFSET
    
    return total_ms

def parse_azure_timeline(file_path):
    """Parse Azure timeline file and return formatted viseme data with compressed silence at the end"""
    viseme_data = []
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Skip the header lines
        timeline_lines = [line for line in lines if line.startswith('Time:')]
        
        # First parse all visemes
        for line in timeline_lines:
            match = re.match(r'Time: ([\d:.]+) - Azure Viseme: (\d+)', line)
            if match:
                time_str, viseme_id = match.groups()
                offset = convert_time_to_offset(time_str)
                
                viseme_data.append({
                    'visemeId': int(viseme_id),
                    'offset': float(offset)
                })

        # Process consecutive silence visemes from the end
        if viseme_data:
            # Start from the end and count consecutive silence visemes
            silence_count = 0
            for i in range(len(viseme_data) - 1, -1, -1):
                if viseme_data[i]['visemeId'] == 0:
                    silence_count += 1
                else:
                    break

            # If we have more than 1 silence viseme at the end
            if silence_count > 1:
                # Keep only the first silence viseme and remove the rest
                viseme_data = viseme_data[:-silence_count + 1]

        return viseme_data
    except Exception as e:
        print(f"Error parsing timeline file: {str(e)}")
        return None

def melSpectra(y, sr, wsize=0.04, hsize=0.04):
    """Calculate mel spectrogram with the same parameters as original script"""
    winLength = int(wsize * sr)
    hopLength = int(hsize * sr)
    
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=64,
        n_fft=winLength, hop_length=hopLength,
        win_length=winLength, window='hann'
    )
    
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

class AudioProcessor:
    def __init__(self, model_path, output_dir):
        self.model = self.load_or_create_model(model_path)
        self.output_dir = output_dir
        self.azure_viseme_map = {
            "Silence": [0], "P/B/M (Bilabial)": [21],
            "F/V (Labiodental)": [18], "TH (Dental)": [17, 19],
            "T/D/N/L (Alveolar)": [14, 19], "S/Z (Sibilant)": [15],
            "SH/CH/JH/ZH (Palato-alveolar)": [16], "K/G/NG (Velar)": [20],
            "H (Glottal)": [12], "R/W (Labio-velar)": [7, 13],
            "Y (Palatal)": [6], "EY/AE/AH (Vowel)": [1, 2, 4, 5, 6, 8],
            "Rest Position": [0]
        }

    def process_audio_chunk(self, sound, sr, start_idx, chunk_size, wsize=0.04, hsize=0.04):
        """Process audio chunk with proper feature dimensionality"""
        end_idx = min(start_idx + chunk_size, len(sound))
        chunk = sound[start_idx:end_idx]
        
        mel_frames = melSpectra(chunk, sr, wsize, hsize)
        mel_frames = mel_frames.T
        
        zero_vec_d = np.zeros((1, 64), dtype='float32')
        zero_vec_dd = np.zeros((2, 64), dtype='float32')
        
        mel_delta = np.diff(mel_frames, n=1, axis=0)
        mel_delta = np.vstack([zero_vec_d, mel_delta])
        
        mel_ddelta = np.diff(mel_frames, n=2, axis=0)
        mel_ddelta = np.vstack([zero_vec_dd, mel_ddelta])
        
        features = np.concatenate((mel_delta, mel_ddelta), axis=1)
        return self.add_context(features, 5)

    def add_context(self, features, ctx_win):
        """Add context to features"""
        ctx = features.copy()
        filler = features[0, :]
        for i in range(ctx_win):
            features = np.insert(features, 0, filler, axis=0)[:ctx.shape[0], :]
            ctx = np.append(ctx, features, axis=1)
        return ctx

    def extract_audio_features(self, file_path, chunk_frames=75, fs=48000):
        """Extract audio features with proper dimensionality"""
        sound, sr = librosa.load(file_path, sr=fs)
        
        samples_per_frame = int(0.04 * sr)
        total_frames = len(sound) // samples_per_frame
        num_chunks = (total_frames + chunk_frames - 1) // chunk_frames
        
        all_features = []
        for i in range(num_chunks):
            start_sample = i * chunk_frames * samples_per_frame
            chunk_features = self.process_audio_chunk(sound, sr, start_sample, chunk_frames * samples_per_frame)
            if len(chunk_features) > 0:
                all_features.append(chunk_features)
        
        return np.vstack(all_features)

    def load_or_create_model(self, model_path):
        """Load existing model or create new one"""
        try:
            return load_model(model_path)
        except:
            h_dim = 256
            net_in = Input(shape=(75, 768))
            lstm1 = LSTM(h_dim, activation='sigmoid', dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(net_in)
            lstm2 = LSTM(h_dim, activation='sigmoid', dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(lstm1)
            lstm3 = LSTM(h_dim, activation='sigmoid', dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(lstm2)
            dropout = Dropout(0.5)(lstm3)
            l1 = Dense(128, activation='relu')(dropout)
            out = Dense(13, activation='softmax')(l1)
            model = Model(inputs=net_in, outputs=out)
            model.load_weights(model_path)
            return model

    def predict_visemes(self, features, chunk_size=75):
        """Generate viseme predictions in chunks"""
        predictions = []
        for i in range(0, len(features), chunk_size):
            chunk = features[i:i + chunk_size]
            if len(chunk) < chunk_size:
                padding = np.zeros((chunk_size - len(chunk), features.shape[1]))
                chunk = np.vstack([chunk, padding])
            pred = self.model.predict(np.expand_dims(chunk, 0), verbose=0)
            predictions.append(pred[0][:len(chunk)])
        return np.vstack(predictions)

    def generate_timeline(self, predictions, output_file):
        """Generate viseme timeline with timestamps"""
        with open(output_file, 'w') as f:
            for frame_num, frame_pred in enumerate(predictions):
                viseme_id = np.argmax(frame_pred)
                confidence = frame_pred[viseme_id] * 100
                timestamp = str(timedelta(milliseconds=frame_num * 40))
                viseme_name = VISEME_LABELS[viseme_id]
                f.write(f"Time: {timestamp} - Viseme: {viseme_name} (Confidence: {confidence:.2f}%)\n")
        return output_file

    def convert_to_azure(self, timeline_file, output_file):
        """Convert viseme timeline to Azure format"""
        with open(timeline_file, 'r') as f:
            lines = f.readlines()

        with open(output_file, 'w') as f:
            f.write("# Azure Viseme Timeline\n\n")
            for line in lines:
                match = re.match(r'Time: (\d+:\d+:\d+(?:\.\d+)?) - Viseme: (.*?) \(Confidence: (\d+\.\d+)%\)', line)
                if match:
                    time_str, viseme, confidence = match.groups()
                    azure_id = self.azure_viseme_map.get(viseme, [0])[0]
                    f.write(f"Time: {time_str} - Azure Viseme: {azure_id} [Confidence: {float(confidence):.1f}%]\n")

    def process_file(self, file_path):
        """Process a single file with timing information"""
        timings = {}
        
        # Feature extraction
        start_time = time.time()
        features = self.extract_audio_features(file_path)
        timings['feature_extraction'] = time.time() - start_time
        
        # Model prediction
        start_time = time.time()
        predictions = self.predict_visemes(features)
        timings['prediction'] = time.time() - start_time
        
        # Timeline generation
        start_time = time.time()
        basename = Path(file_path).stem
        timeline_file = os.path.join(self.output_dir, f"{basename}_viseme_timeline.txt")
        azure_file = os.path.join(self.output_dir, f"{basename}_azure_viseme_timeline.txt")
        
        self.generate_timeline(predictions, timeline_file)
        self.convert_to_azure(timeline_file, azure_file)
        
        # Parse the azure timeline and get viseme data
        viseme_data = parse_azure_timeline(azure_file)
        
        timings['timeline_generation'] = time.time() - start_time
        return timings, viseme_data

class FileProcessor(FileSystemEventHandler):
    def __init__(self, input_dir, model_path, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.processor = AudioProcessor(model_path, output_dir)
        self.queue = Queue()
        self.results = {}  # Store results for each file
        self.processing_thread = threading.Thread(target=self.process_queue, daemon=True)
        self.processing_thread.start()

    def on_created(self, event):
        if event.src_path.lower().endswith('.wav'):
            self.queue.put(event.src_path)
            print(f"\nNew file detected: {event.src_path}")
            print(f"Current queue size: {self.queue.qsize()}")

    def process_queue(self):
        while True:
            if not self.queue.empty():
                file_path = self.queue.get()
                print(f"\nProcessing: {file_path}")
                
                try:
                    time.sleep(1)
                    
                    start_time = time.time()
                    timings, viseme_data = self.processor.process_file(file_path)
                    total_time = time.time() - start_time
                    
                    print("\nProcessing Times:")
                    print(f"Feature Extraction: {timings['feature_extraction']:.2f}s")
                    print(f"Model Prediction: {timings['prediction']:.2f}s")
                    print(f"Timeline Generation: {timings['timeline_generation']:.2f}s")
                    print(f"Total Processing Time: {total_time:.2f}s")
                    print(f"Remaining files in queue: {self.queue.qsize()}")
                    
                    # Store the results
                    basename = Path(file_path).stem
                    self.results[basename] = viseme_data
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                finally:
                    # Clean up file even if there's an error
                    if os.path.exists(file_path):
                        os.remove(file_path)
                
                self.queue.task_done()
            time.sleep(0.1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize the processor
file_processor = FileProcessor(UPLOAD_FOLDER, MODEL_PATH, PROCESSED_FOLDER)

# Start the file observer
observer = Observer()
observer.schedule(file_processor, UPLOAD_FOLDER, recursive=False)
observer.start()

@app.route('/process-audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        try:
            file.save(filepath)
            
            # Wait for processing to complete
            basename = Path(filename).stem
            azure_file = os.path.join(PROCESSED_FOLDER, f"{basename}_azure_viseme_timeline.txt")
            timeline_file = os.path.join(PROCESSED_FOLDER, f"{basename}_viseme_timeline.txt")
            
            # Wait for file to be processed (with timeout)
            timeout = 2  # seconds
            start_time = time.time()
            while not os.path.exists(azure_file) and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if os.path.exists(azure_file):
                # Get viseme data
                viseme_data = parse_azure_timeline(azure_file)
                
                # Clean up all files
                if os.path.exists(filepath):
                    print("remove")
                    #os.remove(filepath)
                if os.path.exists(azure_file):
                    os.remove(azure_file)
                if os.path.exists(timeline_file):
                    os.remove(timeline_file)
                
                return jsonify({
                    'message': 'Audio processed successfully',
                    'filename': filename,
                    'visemes': viseme_data
                }), 200
            else:
                # Clean up audio file if processing failed
                if os.path.exists(filepath):
                    print("remove")
                    #os.remove(filepath)
                return jsonify({
                    'error': 'Processing timeout'
                }), 408
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            if os.path.exists(azure_file):
                os.remove(azure_file)
            if os.path.exists(timeline_file):
                os.remove(timeline_file)
                
            return jsonify({
                'error': f'Error processing file: {str(e)}'
            }), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        'queue_size': file_processor.queue.qsize(),
        'upload_directory': UPLOAD_FOLDER,
        'processed_directory': PROCESSED_FOLDER
    })

if __name__ == '__main__':
    try:
        print(f"Server starting...")
        print(f"Monitoring directory: {UPLOAD_FOLDER}")
        print(f"Output directory: {PROCESSED_FOLDER}")
        print("Waiting for WAV files...")
        app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        observer.stop()
        observer.join()