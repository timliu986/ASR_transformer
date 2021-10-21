import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt


train_label_data = pd.read_csv('E:\\Dataset\\中文語音資料\\zh-TW\\train.tsv',sep = '\t')
test_label_data = pd.read_csv('E:\\Dataset\\中文語音資料\\zh-TW\\test.tsv',sep = '\t')
valid_label_data = pd.read_csv('E:\\Dataset\\中文語音資料\\zh-TW\\validated.tsv',sep = '\t')
valid_label_data['sentence'].str.len()
p = valid_label_data.iloc[40678,:]['path']
y,sr = librosa.load('E:\\Dataset\\中文語音資料\\zh-TW\\clips\\%s'%p, sr=None)

chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
chroma_med = librosa.decompose.nn_filter(chroma,
                                         aggregate=np.median,
                                         metric='cosine')

plt.plot(y)
plt.show()


mfccs = librosa.feature.mfcc(y=yy, sr=sr, n_mfcc=39, hop_length=int(sr / 100), n_fft=int(sr / 40))
mfccs.shape
plt.plot(yy)
plt.show()
len(yy)

def cut_head_tail(sequnce,width = 5000,threshold = 0.01):  # stop 1:star ,2: end
    d = np.copy(sequnce)
    i = 0
    stop = 0
    while np.sum(np.square(d[i:i + width])) < threshold:
        i += width
        if i > len(d):
            print('cut error on head')
            stop = 1
            break
    if stop == 1:
        return d ,stop
    else:
        t = len(d)
        while np.sum(np.square(d[t-width:t])) < threshold:
            t -= width
            if t < 0:
                print('cut error on tail')
                stop = 2
                break
        if stop == 2:
            return d[i:] ,stop
        else :
            output = d[i:t]
            return output ,stop



'''
p = train_label_data['path'][153]
p = train_label_data['path'][155]
p = test_label_data.iloc[27,:]['path']
d = test_label_data.iloc[27,:]['client_id']
train_label_data['path'] in valid_label_data['path']
valid_label_data[valid_label_data['client_id'] == d]['path']
y,sr = librosa.load('E:\\Dataset\\中文語音資料\\zh-TW\\clips\\%s'%p, sr=None)

plt.plot(y)
plt.title(p)
plt.show()
yy ,stop = cut_head_tail(y)
plt.plot(yy)
plt.title(p+' after cut')
plt.show()



mfccs = librosa.feature.mfcc(y=yy, sr=sr, n_mfcc=39, hop_length=int(sr / 100),
                             n_fft=int(sr / 40))
mel = librosa.feature.melspectrogram(y=yy, sr=sr, n_mels=39, hop_length=int(sr / 100),
                             n_fft=int(sr / 40))
mel.shape
plt.plot(mel.T)
plt.show()
plt.plot(mfccs.T)
plt.show()
mel.shape
mfccs.shape
'''

def read_sound_file(label ,mfcc_features):
    sound_file = []
    rate = []
    k = 1
    cut_errpr = {}
    for sound in label['path']:
        y, sr = librosa.load('E:\\Dataset\\中文語音資料\\zh-TW\\clips\\%s' % sound, sr=None)
        y ,stop = cut_head_tail(y)
        if stop > 0:
            cut_errpr.setdefault(sound ,stop)
        else:
            sound_file.append(y)
            rate.append(sr)
            print('had read %s/%s on sound file'%(k,len(label['path'])))
            k += 1
    max_len = len(max(sound_file ,key = len))
    print('-'*50)
    print('max length:',max_len)
    print('-'*50)
    mel = []
    k = 1
    for i in range(len(sound_file)):
        pad_len = max_len-len(sound_file[i])
        sound_file[i] = np.pad(sound_file[i] ,(0,int(pad_len)), mode='constant')
        #mfccs = librosa.feature.mfcc(y=sound_file[i], sr=rate[i], n_mfcc=mfcc_features, hop_length=int(rate[i] / 100), n_fft=int(rate[i] / 40))
        mel = librosa.feature.melspectrogram(y=sound_file[i], sr=rate[i], n_mels=39, hop_length=int(rate[i] / 100),
                                             n_fft=int(rate[i] / 40))
        # with window width 25ms ,stride 10 ms
        mel.append(mel)
        print('had changed %s/%s to mfcc' % (k,len(sound_file)))
        k += 1
    return sound_file ,mel ,cut_errpr

def read_sound_file2(label ,mfcc_features ,max_sen):
    sound_file = []
    rate = []
    senten = []
    k = 1
    cut_errpr = {}
    for sound,sentence in zip(label['path'],label['sentence']):
        if len(sentence)<max_sen:
            y, sr = librosa.load('E:\\Dataset\\中文語音資料\\zh-TW\\clips\\%s' % sound, sr=None)
            y ,stop = cut_head_tail(y)
            if stop > 0:
                cut_errpr.setdefault(sound ,stop)
            elif len(y)<150000:
                sound_file.append(y)
                rate.append(sr)
                senten.append(sentence)
            else :
                del y,sr,stop
        else:
            continue
        print('had read %s/%s on sound file'%(k,len(label['path'])))
        k += 1
    max_len = len(max(sound_file ,key = len))
    max_ind = sound_file.index(max(sound_file ,key = len))
    mfcc = []
    k = 1
    length = len(sound_file)
    for i in range(length):
        pad_len = max_len-len(sound_file[-1])
        sound_file[-1] = np.pad(sound_file[-1] ,(0,int(pad_len)), mode='constant')
        mfccs = librosa.feature.mfcc(y=sound_file[-1], sr=rate[-1], n_mfcc=mfcc_features, hop_length=int(rate[-1] / 100), n_fft=int(rate[-1] / 40))
        mfcc.append(mfccs)

        print('had changed %s/%s to mfcc' % (k,length))
        k += 1
        del sound_file[-1]
        del rate[-1]
    print('-'*50)
    print('max length:',max_len)
    print('file number:',len(mfcc))
    print('max index:',max_ind)
    print('-'*50)

    return mfcc ,cut_errpr ,senten

def read_sound_file3(label ,mel_features):
    sound_file = []
    rate = []
    k = 1
    cut_errpr = {}
    for sound in label['path']:
        y, sr = librosa.load('E:\\Dataset\\中文語音資料\\zh-TW\\clips\\%s' % sound, sr=None)
        y ,stop = cut_head_tail(y)
        if stop > 0:
            cut_errpr.setdefault(sound ,stop)
        else:
            sound_file.append(y)
            rate.append(sr)
            print('had read %s/%s on sound file'%(k,len(label['path'])))
            k += 1
    max_len = len(max(sound_file ,key = len))
    print('-'*50)
    print('max length:',max_len)
    print('-'*50)
    mel = []
    k = 1
    length = len(sound_file)
    for i in range(length):
        pad_len = max_len-len(sound_file[-1])
        sound_file[-1] = np.pad(sound_file[-1] ,(0,int(pad_len)), mode='constant')
        mels = librosa.feature.melspectrogram(y=sound_file[-1], sr=rate[-1], n_mels=mel_features, hop_length=int(rate[-1] / 100),
                                             n_fft=int(rate[-1] / 40))
        # with window width 25ms ,stride 10 ms
        mel.append(mels)

        print('had changed %s/%s to melspectrogram' % (k,length))
        k += 1
        del sound_file[-1]
        del rate[-1]

    return mel ,cut_errpr


train_mfcc,train_error ,sentence = read_sound_file2(train_label_data ,mfcc_features = 39 ,max_sen=10)##         mfcc:39,723
len(train_mfcc)
np.save('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\train_mfcc_reduce',train_mfcc)
train_label = train_label_data[train_label_data['sentence'].isin(sentence)]['sentence'].reset_index(drop=True).str.join(',').str.split(',')
np.save('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\train_label_reduce',train_label)
del train_mfcc ,train_error ,sentence
test_mfcc ,test_error,sentence = read_sound_file2(test_label_data,mfcc_features = 39 ,max_sen=10)##         mfcc:39,1030
np.save('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\test_mfcc_reduce',test_mfcc)
test_label = test_label_data[test_label_data['sentence'].isin(sentence)]['sentence'].reset_index(drop=True).str.join(',').str.split(',')
np.save('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\test_label_reduce',test_label)
del test_mfcc ,test_error ,sentence

valid_mfcc ,valid_error,sentence = read_sound_file2(valid_label_data,mfcc_features = 39 ,max_sen=10)##         mfcc:39,1030
np.save('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\valid_mfcc_reduce',valid_mfcc)
valid_label = valid_label_data[valid_label_data['sentence'].isin(sentence)]['sentence'].reset_index(drop=True).str.join(',').str.split(',')
np.save('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\valid_label_reduce',valid_label)
del valid_mfcc ,valid_error ,sentence

np.save('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\test_mfcc_80',test_mfcc)
del test_mfcc ,test_error
valid_mfcc ,valid_error = read_sound_file2(valid_label_data,mfcc_features = 39 ,max_len = 10)
len(valid_mfcc)
np.save('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\valid_mfcc_reduce',valid_mfcc)
np.save('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\valid_mfcc_80_2',valid_mfcc[7000:14000])
np.save('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\valid_mfcc_80_3',valid_mfcc[14000:21000])
np.save('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\valid_mfcc_80_4',valid_mfcc[21000:28000])
np.save('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\valid_mfcc_80_5',valid_mfcc[28000:35000])
np.save('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\valid_mfcc_80_6',valid_mfcc[35000:])
del valid_mfcc ,valid_error
len(valid_mfcc)


train_mel,train_error = read_sound_file3(train_label_data ,mel_features = 80) ##         mfcc:39,723
test_mel ,test_error = read_sound_file3(test_label_data,mel_features = 80)##         mfcc:39,1030
valid_mel ,valid_error = read_sound_file3(valid_label_data,mel_features = 80)
train_label = train_label_data['sentence'].str.join(',').str.split(',')
test_label = test_label_data['sentence'].str.join(',').str.split(',')
valid_label = valid_label_data['sentence'].str.join(',').str.split(',')


np.save('train_sound_file',train_sound_file)
np.save('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\train_mfcc_80',train_mfcc)
np.save('test_sound_file',test_sound_file)
np.save('test_mfcc',test_mfcc)
np.save('valid_mfcc',valid_mfcc)
np.save('train_label',train_label)
np.save('test_label',test_label)
np.save('valid_label',valid_label)

len(valid_mel)
np.save('train_mel' ,train_mel)
np.save('test_mel' ,test_mel)
np.save('valid_mel_1',valid_mel[:20340])
np.save('valid_mel_2',valid_mel)
valid_mel = valid_mel[20340:]
del test_error
del train_mel
del test_mel
print('File saved successfully')