import librosa
import numpy as np

def predict_model(y, sr, model, labels, window_size=1.0, stride=0.5, ):
    duration = librosa.get_duration(y=y, sr=sr)
    predictions = []
    timestamps = []

    for start in np.arange(0, duration - window_size + stride, stride):
        end = start + window_size
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        y_segment = y[start_sample:end_sample]

        if len(y_segment) < sr * window_size:
            padding = sr * window_size - len(y_segment)
            y_segment = np.pad(y_segment, (0, int(padding)))

        mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

        prediction = model.predict(mfcc_mean)
        predicted_class = np.argmax(prediction)
        predictions.append(predicted_class)
        timestamps.append((round(start, 2), round(end, 2)))

    return [(f"{start}-{end} detik", labels[pred]) for (start, end), pred in zip(timestamps, predictions)]