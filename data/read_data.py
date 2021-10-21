import pandas as pd
import librosa
import numpy as np
import os
import pydub
import matplotlib.pyplot as plt

loc = 'E:\\Dataset\\THCHS-30'
train_list = [d for d in os.listdir(os.path.join(loc,'train')) if d[-3:] == 'wav' ]
train_label_list = [d for d in os.listdir(os.path.join(loc,'train')) if d[-3:] == 'trn' ]




train_label_data = pd.read_csv(os.path.join(loc,'train'),sep = '\t')
test_label_data = pd.read_csv(os.path.join(loc,'test.tsv'),sep = '\t')
valid_label_data = pd.read_csv(os.path.join(loc,'validated.tsv'),sep = '\t')

speech_loc = os.path.join(loc,'clips')
p = train_label_data.iloc[45,:]['path']

y,sr = librosa.load("E:\\Dataset\\THCHS-30\\train\\A2_0.wav")
plt.plot(y)
