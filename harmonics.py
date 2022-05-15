import librosa
import scipy.signal as signal
import numpy as np


audio_sample, sampling_rate = librosa.load("akmu.wav", sr = None)

S = np.abs(librosa.stft(audio_sample, n_fft=1024, hop_length=512, win_length = 1024, window=signal.hann))
pitches, magnitudes = librosa.piptrack(S=S, sr=sampling_rate)

shape = np.shape(pitches)
nb_samples = shape[0]
nb_windows = shape[1]

for i in range(0, nb_windows):
    index = magnitudes[:,i].argmax()
    pitch = pitches[index,i]
    print(pitch)

# FFT 결과를 plot
import matplotlib.pyplot as plt
import librosa.display

D = librosa.feature.melspectrogram(S=S, sr=sampling_rate)
D_dB = librosa.power_to_db(D, ref=np.max)


#하모닉스 추출
harm = librosa.effects.harmonic(D_dB, margin=1.0)
plt.figure(figsize=(12, 4))
librosa.display.specshow(harm, y_axis='mel', x_axis='time', sr=sampling_rate)
plt.title('harmonic spectogram')
plt.colorbar()
plt.tight_layout()
plt.show()