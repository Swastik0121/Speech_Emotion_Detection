import pyaudio
import wave
from flask import Flask, request, render_template
app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 3
    OUTPUT_FILENAME = "input_audio.wav"
    
    audio = pyaudio.PyAudio()
    
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                       rate=RATE, input=True,
                      frames_per_buffer=CHUNK)
    
    print("Recording...")
    
    frames = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("Finished recording.")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    wf = wave.open(OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    #########################################model trained####################################
    import pandas as pd
    import numpy as np
    import os
    import sys
    import joblib
    
    from tensorflow.keras.models import load_model
    import matplotlib.pyplot as plt
     
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.model_selection import train_test_split
    
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    
    # load model
    model = load_model('trained_model.h5')
    # summarize model.
    #model.summary()
    
    #######################################for extraction########################################
    
    import librosa
    import librosa.display
    from IPython.display import Audio
    
    #############module for prediction#################################################
    
    #path = 'OAF_back_fear.wav'
    path = 'input_audio.wav'
    #path = 'OAF_chalk_angry.wav'
    data, sample_rate = librosa.load(path)
    
    #adding different types of noises to augment the pure data
    
    
    #adds white noise to the data
    def noise(data):
        noise_amplitude = 0.01*np.random.uniform()*np.amax(data)
        data = data + noise_amplitude*np.random.normal(size = data.shape[0])
        return data
    
    
    #stretchs the audio by few seconds
    def stretch(data):
        return librosa.effects.time_stretch(data, rate= 0.1)
    
    
    #changes the pitch of the audio
    def pitch(data, sampling_rate):
        return librosa.effects.pitch_shift(data, sr = sampling_rate,n_steps = 0.1)
    
    
    def extracted_features(data):
        
        #zcr = detects sudden change in the signal rate at which the signal changes rate
        result = np.array([])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y = data).T, axis = 0)
        result = np.hstack((result, zcr)) # horizontally stacking
        # also dtransposing to make each column correspond to a frame 
        # and each row correspond to a feature (zero-crossing rate).
        
        # chroma_stft = splits audio in 12 different overlapping classes and applies stft.
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S = stft, sr = sample_rate).T, axis = 0)
        result = np.hstack((result, chroma_stft))
        
        # mfcc =  derived from the mel scale, which is a perceptual scale of pitches that mimics the human auditory system's response to different frequencies.
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mfcc))
        
        #rms = root mean square
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        result = np.hstack((result, rms))
        
        #mel_spectrogram = Mel spectrogram is a visual representation of audio signals in the frequency domain.
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
        
        return result
    
    def get_features(data):
        
        data, sample_rate = librosa.load(path, duration = 2.5, offset = 0.2)
        
        #pure data
        res1 = extracted_features(data)
        result = np.array(res1)
        
    
        #adding noise
        noise_data = noise(data)
        res2 = extracted_features(noise_data)
        result = np.vstack((result, res2))
        
        #stretching and pitching the data
        new_data = stretch(data)
        pitch_data = pitch(new_data, sample_rate)
        res3 = extracted_features(pitch_data)
        result = np.vstack((result, res3))
    
        
        return result
    
    X__ = []
    feature = get_features(path)
    for ele in feature:
        X__.append(ele)   
        
    scaler = StandardScaler()
    encoder = joblib.load("encoder_emotion.pkl")
    
    # Apply StandardScaler to the list of features
    X__ = scaler.fit_transform(X__).reshape(1,-1)
    X__ = X__[:, :162]
    test_input = np.expand_dims(X__, axis=2)
    
    pred_test__ = model.predict(test_input)
    y_pred__ = encoder.inverse_transform(pred_test__)
    y_pred_string = np.array_str(y_pred__)
    
    #######################
    print(y_pred_string)
    return render_template('index2.html',x=y_pred_string)

@app.route('/redirect', methods=['POST'])
def redirect():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=False)