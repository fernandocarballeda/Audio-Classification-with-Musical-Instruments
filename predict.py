from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
from collections import Counter
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pandas as pd
import librosa
from librosa.core import resample, to_mono
from tqdm import tqdm
import wavio

def load_audio(file_path):
    # Carga el archivo de audio y devuelve la forma mono y la frecuencia de muestreo
    wav, sr = librosa.load(file_path, sr=None, mono=True)
    return wav

def has_sound(file_path, threshold=0.01):
    y, sr = librosa.load(file_path)
    rms = librosa.feature.rms(y=y)
    return rms.max() > threshold

def delete_silent_files(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".wav"):
                file_path = os.path.join(root, filename)
                if not has_sound(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        mask.append(mean > threshold)
    return mask, y_mean

def downsample_mono(path, sr):
    obj = wavio.read(path)
    wav = obj.data.astype(np.float32, order='F')
    rate = obj.rate
    try:
        channel = wav.shape[1]
        if channel == 2:
            wav = to_mono(wav.T)
        elif channel == 1:
            wav = to_mono(wav.reshape(-1))
    except IndexError:
        wav = to_mono(wav.reshape(-1))
    except Exception as exc:
        raise exc
    wav = resample(wav, rate, sr)
    wav = wav.astype(np.int16)
    return sr, wav

def save_sample(sample, rate, target_dir, fn, ix):
    fn = fn.split('.wav')[0]
    dst_path = os.path.join(target_dir.split('.')[0], fn+'_{}.wav'.format(str(ix)))
    if not os.path.exists(dst_path):
        wavfile.write(dst_path, rate, sample)

def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def split_wavs_single(args):
    predict_dir = args.pred_dir
    dst_root = args.dst_root
    dt = args.delta_time

    wav_paths = glob('{}/*.wav'.format(predict_dir))
    wav_paths = [x for x in wav_paths if '.wav' in x]
    check_dir(dst_root)

    for src_fn in wav_paths:
        target_dir = dst_root
        check_dir(target_dir)

        rate, wav = downsample_mono(src_fn, args.sr)
        mask, y_mean = envelope(wav, rate, threshold=args.threshold)
        wav = wav[mask]
        delta_sample = int(dt * rate)

        if wav.shape[0] < delta_sample:
            sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
            sample[:wav.shape[0]] = wav
            save_sample(sample, rate, target_dir, args.fn, 0)
        else:
            trunc = wav.shape[0] % delta_sample
            for cnt, i in enumerate(np.arange(0, wav.shape[0]-trunc, delta_sample)):
                start = int(i)
                stop = int(i + delta_sample)
                sample = wav[start:stop]
                save_sample(sample, rate, target_dir, args.fn, cnt)

    delete_silent_files(dst_root)

def make_prediction(args):
    model = load_model(args.model_fn,
        custom_objects={'STFT':STFT,
                        'Magnitude':Magnitude,
                        'ApplyFilterbank':ApplyFilterbank,
                        'MagnitudeToDecibel':MagnitudeToDecibel})
    wav_paths = glob('{}/**/*.wav'.format(args.pred_dir_cleaned), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths])
    classes = sorted(os.listdir(args.src_dir))
    le = LabelEncoder()
    le.fit(classes)
    
    predictions = []
    global_predictions = []
    y_pred_list = []

    for wav_fn in tqdm(wav_paths, total=len(wav_paths)):
        wav = load_audio(wav_fn)  # Cargar el archivo de audio
        wav = np.expand_dims(wav, axis=0)  # Añadir dimensión de lote si es necesario
        y_pred = model.predict(wav)  # Predecir la clase
        y_pred_list.append(y_pred[0])  # Almacenar la predicción del segmento

        y_pred_global = np.argmax(y_pred[0]) 
        print(y_pred_global) # Obtener la clase predicha global
        predicted_class_global = classes[y_pred_global]  # Obtener el nombre de la clase
        global_predictions.append(predicted_class_global)

        # Imprimir la predicción global para el archivo actual
        print(f'Predicted class of {wav_fn}: {predicted_class_global}')
        predictions.append((wav_fn, predicted_class_global))

    # Convert y_pred_list to numpy array for easier manipulation
    y_pred_array = np.array(y_pred_list)

    # Calcular la probabilidad media para cada clase
    mean_y_pred = np.mean(y_pred_array, axis=0)
    print(f'Mean prediction for each class: {mean_y_pred}')

    # Determine the class with the highest mean probability
    mean_pred_class = classes[np.argmax(mean_y_pred)]
    print(f'Mean prediction class: {mean_pred_class}')

    # Determine the most frequent predicted class
    global_prediction = Counter(global_predictions).most_common(1)[0][0]
    print(f'Global prediction is: {global_prediction}')

    return predictions, global_prediction, mean_y_pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--src_root', type=str, default='predict_dir',
                        help='directory of audio files in total duration')
    parser.add_argument('--classes_dir', type=str, default='wavfiles',
                        help='directory of audio classes')
    parser.add_argument('--model_fn', type=str, default='models_folder/conv2d.h5',
                        help='model file to make predictions')
    parser.add_argument('--pred_dir', type=str, default='predict_folder',
                        help='predict directory containing wavfiles to predict')
    parser.add_argument('--pred_fn', type=str, default='y_pred',
                        help='fn to write predictions in logs dir')
    parser.add_argument('--src_dir', type=str, default='wavfiles',
                        help='directory containing wavfiles to predict')
    parser.add_argument('--dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='sample rate of clean audio')
    parser.add_argument('--threshold', type=float, default=20.0,
                        help='threshold magnitude for np.int16 dtype')
    parser.add_argument('--pred_dir_cleaned', type=str, default='predict_dir_cleaned',
                        help='directory to put audio files split by delta_time')
    parser.add_argument('--dst_root', type=str, default='predict_dir_cleaned',
                        help='directory to put audio files split by delta_time')
    parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--fn', type=str, default='3a3d0279',
                        help='file to plot over time to check magnitude')
    args, _ = parser.parse_known_args()

    split_wavs_single(args)
    make_prediction(args)
