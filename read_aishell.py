import pandas as pd
import librosa
import numpy as np
import os
import  matplotlib.pyplot as plt

label = pd.read_table('E:\\Dataset\\data_aishell\\data_aishell\\transcript\\aishell_transcript_v0.8.txt',sep = '\n',header=None)

new_label = pd.DataFrame({'ID':label.iloc[:,0].str.split(' ').str.get(0),
                          'sentence':pd.concat([label.iloc[:120098,0].str.split(' ').str[1:],label.iloc[120098:,0].str.split('  ').str[2:]],axis = 0)})
del label
new_label.insert(2,'name',new_label['ID'].str[6:11])

train_name = os.listdir('E:\\Dataset\\data_aishell\\data_aishell\\wav\\train')
test_name = os.listdir('E:\\Dataset\\data_aishell\\data_aishell\\wav\\test')
dev_name = os.listdir('E:\\Dataset\\data_aishell\\data_aishell\\wav\\dev')

new_label['sentence'].str.len().describe()
new_label[new_label['sentence'].str.len()<= 11]
p = new_label[new_label['sentence'].str.len()== 11]['name'].iloc[11]
idd = new_label[new_label['sentence'].str.len()== 11]['ID'].iloc[11]

def read_mfcc(data,type):
    mfcc = np.zeros([data.shape[0],39,751])
    error = []
    sente = pd.DataFrame(columns=['sentence'])
    s = 0
    e = 0
    for file ,name ,senc in zip(data['ID'],data['name'],data['sentence']):
        yy = np.zeros([120000])
        y, sr = librosa.load('E:\\Dataset\\data_aishell\\data_aishell\\wav\\%s\\%s\\%s.wav' % (type ,name, file), sr=None)
        print('read  %s/%s file(ss / err) with %s data' %(s ,e, len(data)))
        try:
            yy[:len(y)] = y
            mfccs = librosa.feature.mfcc(y=yy, sr=sr, n_mfcc=39, hop_length=int(sr / 100), n_fft=int(sr / 40))
            mfcc[s] = mfccs
            sente = sente.append(pd.DataFrame({'sentence': [senc]})).reset_index(drop=True)
            s += 1
        except:
            print(name,file,'Length error')
            error.append([y, sr])
            e += 1
        print('-'*20,(s+e),'-'*20)
    mfcc = mfcc[:s-1]
    return mfcc,sente,error

train_mfcc,train_label,train_error = read_mfcc(new_label[new_label['sentence'].str.len()<= 10][new_label['name'].isin(train_name)])
train_mfcc = train_mfcc[:85158]
np.save('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\aishell_train_mfcc',train_mfcc)
np.save('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\aishell_train_label',train_label)
del train_mfcc,train_label


test_mfcc,test_label,test_error = read_mfcc(new_label[new_label['sentence'].str.len()<= 10][new_label['name'].isin(test_name)],type = 'test')

np.save('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\aishell_test_mfcc',test_mfcc)
np.save('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\aishell_test_label',test_label)
del train_mfcc,train_label
## teat error
p = 'S0002'
idd = 'BAC009S0002W0122'
new_label[new_label['ID'] == idd]
yy = np.zeros([140000])
y,sr = librosa.load('E:\\Dataset\\data_aishell\\data_aishell\\wav\\train\\%s\\%s.wav'%(p,idd), sr=None)
yy[:len(y)] = y
y.shape
plt.plot(y)
plt.plot(yy)
plt.show()
mfccs = librosa.feature.mfcc(y=yy, sr=sr, n_mfcc=39, hop_length=int(sr / 100), n_fft=int(sr / 40))
mfccs.shape
plt.plot(mfccs.T)
plt.show()

data = new_label[new_label['sentence'].str.len()<= 11][new_label['name'].isin(train_name)].iloc[:20,:]
mfcc = np.zeros([data.shape[0],39,688])
error = []
s = 0
e = 0
for file ,name in zip(data['ID'],data['name']):
    yy = np.zeros([120000])
    y, sr = librosa.load('E:\\Dataset\\data_aishell\\data_aishell\\wav\\train\\%s\\%s.wav' % (name, file), sr=None)
    print('read  %s/%s file(ss / err) with %s data' %(s ,e, len(data)))
    try:
        yy[:len(y)] = y
        mfccs = librosa.feature.mfcc(y=yy, sr=sr, n_mfcc=39, hop_length=int(sr / 100), n_fft=int(sr / 40))
        mfcc[s] = mfccs
        s += 1
    except:
        print(name,file,'Length error')
        error.append([y, sr])
        e += 1