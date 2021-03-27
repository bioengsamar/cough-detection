from pathlib import Path
import os
import librosa
import numpy as np
import soundfile as sf

def load_audio_file(file_path):
    input_length = 16000
    data = librosa.core.load(file_path)[0] #, sr=16000
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data

    
def Adding_white_noise(data, name):
    wn = np.random.randn(len(data))
    data_wn = data + 0.005*wn
    sf.write('augmented_cough_data/adding_noise/noise_{}'.format(name),data_wn, 16000, 'PCM_24')
    
def Shifting_sound(data, name):
    data_roll = np.roll(data, 1600)
    sf.write('augmented_cough_data/Shifting_sound/shifit_{}'.format(name),data_roll, 16000, 'PCM_24')
    
def stretch_sound(data, rate=1):
    input_length = 16000
    data = librosa.effects.time_stretch(data, rate)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data

def stretch_data(data, name):
    data_stretch =stretch_sound(data, 0.8)
    sf.write('augmented_cough_data/stretch_sound/stretch_{}'.format(name),data_stretch, 16000, 'PCM_24')
    
def iterate_files():
    paths = Path("data/cough").glob('**/*.wav')
    for path in paths:
        path_in_str = str(path)
        Adding_white_noise(load_audio_file(path_in_str), os.path.basename(path_in_str))
        Shifting_sound(load_audio_file(path_in_str), os.path.basename(path_in_str))
        stretch_data(load_audio_file(path_in_str), os.path.basename(path_in_str))
        
if __name__ == "__main__":
    iterate_files()