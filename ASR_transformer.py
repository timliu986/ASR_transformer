import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime
##import read_data

label = np.load('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\valid_label.npy',allow_pickle=True)
label.shape
label2 = np.load('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\train_label.npy',allow_pickle=True)
label2.shape
label3 = np.load('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\test_label.npy',allow_pickle=True)
label3.shape
lable_all = np.concatenate([label,label2,label3])
del label ,label2 ,label3
lable_all.shape
label_all = pd.DataFrame(lable_all)[0]
label_all.apply(lambda x: x.append('<EOS>'))
label_all.apply(lambda x: x.insert(0,'<BOS>'))

wav = np.load('E:\\Dataset\\中文語音資料\\wav_data.npy')
label = np.load('E:\\Dataset\\中文語音資料\\label.npy')
label = pd.DataFrame(label)[0].str.split(' ')
label.apply(lambda x: x.append('<EOS>'))
label.apply(lambda x: x.insert(0,'<BOS>'))
label.str.len().describe()
train_mfcc = np.transpose(train_mfcc,[0,2,1])
test_mfcc = np.transpose(test_mfcc,[0,2,1])



train_label = np.load('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\aishell_train_label.npy',allow_pickle=True)
test_label = np.load('E:\\Dataset\\中文語音資料\\mel_and_mfcc\\aishell_test_label.npy',allow_pickle=True)




train_label =pd.DataFrame(train_label)[0]
test_label =pd.DataFrame(test_label)[0]



label[0]
##  adding  BOS ,EOS
train_label.apply(lambda x: x.append('<EOS>'))
train_label.apply(lambda x: x.insert(0,'<BOS>'))
test_label.apply(lambda x: x.append('<EOS>'))
test_label.apply(lambda x: x.insert(0,'<BOS>'))



def creat_dictionary(input_text):
    find_word = {0:' '}
    word = []
    D = {}
    i = 1
    for line in input_text:
        for w in line:
            if w not in word:
                word.append(w)
                find_word.setdefault(i,w)
                i+=1
                D.setdefault(w,line.count(w))
            else :
                D[w] = D[w]+line.count(w)
    find_idx = dict(zip(find_word.values(),find_word.keys()))
    return D ,find_word,find_idx


D ,D_word,D_idx = creat_dictionary(label_all)
len(D)

def id_word_transfer(input_text,dicitonary ,sentence = True):
    id = []
    if sentence:
        l = []
        for w in input_text:
            l.append(dicitonary[w])
        id.append(l)
    else:
        for line in input_text:
            l = []
            for w in line:
                l.append(dicitonary[w])
            id.append(l)
    return id

id_train = id_word_transfer(label,D_idx,sentence=False)
id_test = id_word_transfer(label,D_idx,sentence=False)

len(max(id_train,key=len))

def padding_mask(inp ,max_len = None):
    if max_len == None:
        max_len = len(max(inp, key=len))
    output = np.zeros([len(inp),max_len])
    i = 0
    for line in inp:
        j = 0
        for word in line:
            output[i , j] = word
            j += 1
        i += 1
    return output

id_train = padding_mask(id_train,max_len = 50)
id_train.shape
id_test = padding_mask(id_test ,max_len = 12)
id_test.shape




'''
Basic speech transformer
Input
input : train_mfcc : [data number ,window number ,mfcc feature]
input length : window number
input_tail : train_mfcc_tail : [data number ,tail index]
label : id_train : [data number ,sentence length] 
label_length : sentence_length
label_dictionary : D
'''

class speech_transformer(object):
    def __init__(self,_input,input_length  ,label ,label_length,test_input ,test_label,end_num,label_dictionary ,dropout_rate = 0.1
                 ):
        self.g = tf.Graph()
        self.input_length = input_length
        self.label_length = label_length
        self.features_num = _input.shape[2]
        self.input = _input
        self.label = label
        self.label_dictionary = label_dictionary
        self.label_dic_size = len(self.label_dictionary)
        self.dropout_rate = dropout_rate
        self.end_num = end_num

        self.test_input = test_input
        self.test_label = test_label
        self.train_cost = []
        self.train_word_acc = []
        self.train_length_acc = []
        self.train_WER = []
        self.test_word_acc = []
        self.test_length_acc = []
        self.test_WER = []

        with self.g.as_default():
            ## build the network:
            self.paramater()
            self.build()
            self.y_hat = self.predict()
            # self.y_hat = tf.cast(tf.argmax(self.pred, axis=2),tf.int32)
            self.word_accuracy ,self.length_accuracy,self.WER = self.accuracy_eval(self.y_hat ,self.tar_real)
            y_label = tf.placeholder(tf.int32, [None, self.label_length-1], name='label_text2')
            self.wa, self.la, self.we = self.accuracy_eval(self.y_hat, y_label)
            self.loss = self.loss_function(self.tar_real, self.pred)
            self.optimizer_(self.loss)
            ## initializer
            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()


        ## create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config ,graph = self.g)
        self.sess.run(self.init_op)
        ##writer = tf.summary.FileWriter("TensorBoard/CNN/4Layers", graph = self.sess.graph)

    def paramater(self):
        self.attention_size = 100
        self.head_num = 4
        self.num_layer =4
        self.e_d_different = 0
        self.weight = {'embedding_wieght':tf.Variable(tf.random_uniform([self.label_dic_size, self.attention_size],-1.0, 1.0),name = 'embedding_weight'),
                           }

    def accuracy_eval(self,sen_input ,target):
        mask = tf.cast(tf.not_equal(0, target), tf.float32)
        word_accuracy = tf.reduce_sum(
            tf.cast(tf.equal(sen_input, target), tf.float32) * mask) / tf.reduce_sum(mask)
        len_acc = tf.equal(tf.reduce_sum(tf.cast(tf.not_equal(0, target), tf.float32), axis=1),
                           tf.reduce_sum(tf.cast(tf.not_equal(0, sen_input), tf.float32), axis=1))
        length_accuracy = tf.reduce_mean(tf.cast(len_acc, tf.float32))
        WER = tf.reduce_mean(tf.edit_distance(tf.contrib.layers.dense_to_sparse(sen_input, eos_token=self.end_num),
                                                   tf.contrib.layers.dense_to_sparse(target, eos_token=self.end_num),
                                                   normalize=True))
        return  word_accuracy ,length_accuracy ,WER

    def input_process(self):
        x = tf.placeholder(tf.float32 ,[None ,self.input_length ,self.features_num],name = 'input_text')
        y = tf.placeholder(tf.int32 ,[None ,self.label_length],name = 'label_text')

        self.is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
        return x ,y

    def get_angles(slef,pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self,position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)

        # apply sin to even indices in the array; 2i
        sines = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        cosines = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, cosines], axis=-1)

        pos_encoding = pos_encoding[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def embedding_layer(self, input,name = None):
        embedding = tf.nn.embedding_lookup(self.weight['embedding_wieght'],input,name = name)
        return embedding

    def mask(self,input):
        mask_seq = tf.cast(tf.equal(input,0),tf.float32)

        return mask_seq[:, tf.newaxis, tf.newaxis, :]

    def look_ahead_mask(self ,size ,name = None):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def self_attention(self ,Q,K,V,mask = None):

        dk = tf.cast(tf.shape(K)[-1] ,tf.float32)
        A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(dk)
        if mask != None:
            A += (mask * -1e9)  # mask 使經過softmax後的padding部分會趨於0
        A = tf.nn.softmax(A)  ##[-1,head_num, q_len, k_len]
        O = tf.matmul(A, V)  ##[batch ,head_num ,sentance_length ,attention_size]


        return O ,A

    def head_concate(self ,O,attention_size ,head_num):
        O = tf.transpose(O, [0, 2, 3, 1]) ##[batch ,sentance_length ,attention_size ,head_num ]
        O_H = tf.reshape(O ,[-1 ,tf.shape(O)[1],attention_size*head_num])
        O_H = tf.layers.dense(O_H,units=attention_size ,activation=None)
        return O_H

    def Muti_attention_layer(self ,Q,K,V,attention_size,head_num ,mask = None):

        ###多維矩陣乘法會將前面維度視為batch，由最後兩維度做矩陣相乘
        Q = tf.layers.dense(Q, units=attention_size * head_num, activation=None)
        Q = tf.reshape(Q, [-1, tf.shape(Q)[1], head_num, attention_size])
        Q = tf.transpose(Q, [0, 2, 1, 3])  ##[-1,head_num, q_len, attention_size ]
        K = tf.layers.dense(K, units=attention_size * head_num, activation=None)
        K = tf.reshape(K, [-1, tf.shape(K)[1], head_num, attention_size])
        K = tf.transpose(K, [0, 2, 1, 3])  ##[-1,head_num, k_len , attention_size ]
        V = tf.layers.dense(V, units=attention_size * head_num, activation=None)
        V = tf.reshape(V, [-1, tf.shape(V)[1], head_num, attention_size])
        V = tf.transpose(V, [0, 2, 1, 3])  ##[-1,head_num, v_len, attention_size ]

        O ,A = self.self_attention(Q,K,V,mask=mask)
        O_H = self.head_concate(O,attention_size,head_num)

        ##[batch ,head_num ,sentance_length ,attention_size]

        return O_H ,A

    def TwoD_attention(self,input ,n,c):

        '''
        1.attention alone time
        '''
        conv_Q_filter = tf.Variable(tf.random_normal([3,3,n,c], stddev=0.01))
        conv_K_filter = tf.Variable(tf.random_normal([3,3,n,c], stddev=0.01))
        conv_V_filter = tf.Variable(tf.random_normal([3,3,n,c], stddev=0.01))
        conv_filter = tf.Variable(tf.random_normal([3,3,n,c], stddev=0.01))
        conv_Q = tf.nn.conv2d(input, conv_Q_filter, strides=[1 ,1 ,1 ,1], padding="SAME")
        conv_Q = tf.transpose(conv_Q,[0, 3, 1, 2])# [batch ,head num ,window num ,mfcc feature]
        conv_K = tf.nn.conv2d(input, conv_K_filter, strides=[1, 1, 1 ,1], padding="SAME")
        conv_K = tf.transpose(conv_K, [0, 3, 1, 2])# [batch ,head num ,window num ,mfcc feature]
        conv_V = tf.nn.conv2d(input, conv_V_filter, strides=[1, 1, 1, 1], padding="SAME")
        conv_V = tf.transpose(conv_V, [0, 3, 1, 2])  # [batch ,head num  ,window num ,mfcc feature]
        conv_inp = tf.nn.conv2d(input, conv_filter, strides=[1, 1, 1, 1], padding="SAME")
        conv_inp = tf.transpose(conv_inp, [0, 3, 1, 2])  # [batch ,head num  ,window num ,mfcc feature]
        O ,A = self.self_attention(conv_Q,conv_K,conv_V,mask=None)
        out = tf.contrib.layers.layer_norm(O + conv_inp)
        out = tf.transpose(out ,[0,2,3,1])

        '''
        concat convalution output
        '''
        out1_filter = tf.Variable(tf.random_normal([3,3,c,c], stddev=0.01))
        out2_filter = tf.Variable(tf.random_normal([3,3,c,c], stddev=0.01))
        out1 = tf.nn.conv2d(out ,out1_filter ,strides = [1,1,1,1] ,padding = 'SAME')
        out1 = tf.layers.batch_normalization(out1)
        out2 = tf.nn.conv2d(out1 ,out2_filter ,strides = [1,1,1,1] ,padding = 'SAME')
        out2 = tf.layers.batch_normalization(out2)
        output = tf.nn.relu(out2 + out)
        ## [batch ,window num ,mfcc feature,n]

        return output ,A

    def feed_foward_network(self,input,hidden_units):
        size = input.shape[-1]
        h1 = tf.layers.dense(input ,units=hidden_units ,activation=tf.nn.relu)
        out = tf.layers.dense(h1 ,units=size ,activation=None)
        return  out

    def EncoderLayer(self ,input ,head_num ,attention_size):
        O, A = self.Muti_attention_layer(input, input, input, attention_size=attention_size, head_num=head_num)
        O = tf.layers.dropout(O, rate=self.dropout_rate, training=self.is_train)
        out1 = tf.contrib.layers.layer_norm(O+input)

        ffout = self.feed_foward_network(out1 ,attention_size*4)
        ffout = tf.layers.dropout(ffout, rate=self.dropout_rate, training=self.is_train)
        out2 = tf.contrib.layers.layer_norm(ffout+out1)
        ''' concate dense'''
        #out2 = tf.transpose(out2 , [0 ,2 ,1])
        #out2 = tf.layers.dense(out2 ,units = attention_size ,activation = None)
        #out2 = tf.transpose(out2 , [0 ,2 ,1])
        return out2

    def DecoderLayer(self ,input,enc_input ,head_num ,attention_size ,combined_mask,inp_mask = None):
        O ,A1= self.Muti_attention_layer(input,input,input, attention_size=attention_size, head_num=head_num, mask=combined_mask)
        O = tf.layers.dropout(O, rate=self.dropout_rate, training=self.is_train)
        ##out1 = tf.contrib.layers.layer_norm(O+input)
        out1 = O+input

        O_1 ,A2 = self.Muti_attention_layer(Q = out1,K = enc_input,V = enc_input, attention_size=attention_size, head_num=head_num, mask=inp_mask)
        ##這裡的mask遮掉的是encoder的padding部分,因此使用encoder 的mask
        O_1 = tf.layers.dropout(O_1, rate=self.dropout_rate, training=self.is_train)
        #out2 = tf.contrib.layers.layer_norm(O_1+out1)
        out2 = O_1+out1

        ffout = self.feed_foward_network(out2, attention_size * 4)
        ffout = tf.layers.dropout(ffout, rate=self.dropout_rate, training=self.is_train)
        #out3 = tf.contrib.layers.layer_norm(ffout + out2)
        ffout = ffout + out2

        return ffout ,A1 ,A2

    def loss_function(self ,real ,pred):
        #loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred)
        #mask = tf.logical_not(tf.equal(real, 0))
        #mask = tf.cast(mask, dtype=loss_.dtype)
        #loss_ *= mask  # 只計算非 <pad> 位置的損失

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        loss_ = loss_object(real, pred)
        self.l = loss_

        mask = tf.logical_not(tf.equal(real, 0))
        mask = tf.cast(mask, dtype = loss_.dtype)
        loss_ *= mask  # 只計算非 <pad> 位置的損失


        return tf.reduce_mean(loss_,name = 'loss')

    def optimizer_(self ,loss):
        global_step = tf.Variable(1 ,dtype = tf.float32, trainable=False)
        warmup_steps = 8000
        arg1 = tf.rsqrt(global_step)
        arg2 = global_step*(warmup_steps**-1.5)
        learning_rate = tf.rsqrt(float(self.attention_size))*tf.minimum(arg1 ,arg2)

        opt = tf.train.AdamOptimizer(learning_rate=learning_rate ,beta1=0.9, beta2=0.98,
                                 epsilon=1e-9).minimize(loss,name = 'train_op')
        self.add_global = global_step.assign_add(1)

    def predict(self):
        y_hat = tf.cast(tf.argmax(self.pred, axis=2), tf.int32)
        c = tf.ones_like(y_hat)[:, 0] * self.end_num
        c = tf.expand_dims(c, 1)
        y_hat_ex = tf.concat([y_hat, c], axis=1)
        ind = tf.where(tf.equal(y_hat_ex, self.end_num))
        ind = tf.segment_min(ind[:, 1], ind[:, 0])+1
        mask = tf.cast(tf.sequence_mask(ind, self.label_length), tf.int32)
        result = (y_hat_ex * mask)[:, :-1]

        return result

    def muti_stream_embedding(self, _input, reduce_num,dilation,stream_num):
        tdnn_op = None
        for stream in range(stream_num):
            conv1_filter = tf.Variable(tf.random_normal([3, self.features_num, self.features_num], stddev=0.01))
            conv1_enc_inp = tf.nn.conv1d(_input, conv1_filter, stride=2 ,dilations = dilation[stream],padding='SAME')
            conv1_enc_inp = tf.layers.batch_normalization(conv1_enc_inp)
            conv2_filter = tf.Variable(tf.random_normal([3, self.features_num, self.features_num], stddev=0.01))
            conv2_enc_inp = tf.nn.conv1d(conv1_enc_inp, conv2_filter,dilations = dilation[stream], stride=2, padding='SAME')
            conv2_enc_inp = tf.layers.batch_normalization(conv2_enc_inp)
            conv3_filter = tf.Variable(tf.random_normal([3, self.features_num, self.features_num], stddev=0.01))
            conv3_enc_inp = tf.nn.conv1d(conv2_enc_inp, conv3_filter,dilations = dilation[stream], stride=2, padding='SAME')
            conv3_enc_inp = tf.layers.batch_normalization(conv3_enc_inp)
            out = tf.reshape(conv3_enc_inp, [-1, reduce_num, self.features_num])
            att_out ,A= self.Muti_attention_layer(out,out,out,attention_size=self.features_num,head_num=self.head_num)
            att_out = tf.contrib.layers.layer_norm(att_out)
            ff_out = self.feed_foward_network(att_out ,hidden_units= self.features_num*4)
            ff_out = tf.contrib.layers.layer_norm(ff_out)
            if tdnn_op == None:
                tdnn_op = ff_out
            else:
                tdnn_op = tf.concat([tdnn_op,ff_out],axis = 2)

        tdnn_out = tf.layers.dense(tdnn_op ,units = self.features_num ,activation = tf.nn.relu)
        tdnn_out = tf.layers.batch_normalization(tdnn_out)
        tdnn_out = tf.layers.dropout(tdnn_out, rate=self.dropout_rate, training=self.is_train)

        return tdnn_out

    def input_embedding(self ,_input,n,reduce_num = 79):
        tdnn_out = self.muti_stream_embedding(_input ,reduce_num ,dilation = [1,2,3] ,stream_num = 3)
        tdnn_out = tdnn_out[:,:,:,tf.newaxis]
        enc_inp_1_twod ,A1= self.TwoD_attention(tdnn_out ,n = 1,c = int(n/2))
        enc_inp_2_twod ,A2= self.TwoD_attention(enc_inp_1_twod ,n = int(n/2) ,c = n)
        out = tf.reshape(enc_inp_2_twod ,[-1 ,reduce_num,self.features_num*n])
        enc_inp_out = tf.layers.dense(out ,units = self.attention_size)
        enc_inp_out = tf.contrib.layers.layer_norm(enc_inp_out)

        enc_inp_out += self.positional_encoding(reduce_num, self.attention_size)
        enc_inp_out = tf.layers.dropout(enc_inp_out, rate=self.dropout_rate, training=self.is_train)
        return enc_inp_out ,A1 ,A2

    def decoder_processes(self ,_input):
        tar_inp = _input[:,:-1]
        tar_real = _input[:,1:]

        dec_inp = self.embedding_layer(tar_inp)
        dec_inp += self.positional_encoding(self.label_length-1, self.attention_size)
        return tar_inp,tar_real,dec_inp

    '''
    def build(self):
        self.x , self.y = self.input_process()
        self.enc_inp ,self.A1 ,self.A2= self.input_embedding(self.x ,n = 32,reduce_num = 205)

        self.tar_inp ,self.tar_real ,self.dec_inp = self.decoder_processes(self.y)
        m = self.mask(self.tar_inp)
        look_ahead_mask = self.look_ahead_mask(m.shape[-1])
        combined_mask = tf.maximum(m, look_ahead_mask)
        self.e = []
        self.d = []
        self.A = []

        for layers in range(self.num_layer):
            try:
                e = self.EncoderLayer(self.e[layers-1] ,head_num = self.head_num ,attention_size = self.attention_size )
                print('encoder %s' % (layers + 1))
                if len(self.e) - len(self.d) == self.e_d_different:
                    try:
                        d, A1, A2 = self.DecoderLayer(self.d[layers - 1], e, head_num=self.head_num,
                                                      attention_size=self.attention_size,
                                                      combined_mask=combined_mask)

                        self.d.append(d)
                        self.A.append([A1,A2])
                        print('decoder %s' % (layers + 1))
                    except:
                        d, A1, A2 = self.DecoderLayer(self.dec_inp, e, head_num=self.head_num,
                                                      attention_size=self.attention_size,
                                                      combined_mask=combined_mask)

                        self.d.append(d)
                        self.A.append([A1, A2])
                        print('decoder %s' % (layers + 1))

                self.e.append(e)

            except:
                e = self.EncoderLayer(self.enc_inp, head_num=self.head_num, attention_size=self.attention_size)
                if self.e_d_different == 0:
                    d, A1, A2 = self.DecoderLayer(self.dec_inp, e, head_num=self.head_num,
                                                  attention_size=self.attention_size,
                                                  combined_mask=combined_mask)

                    self.d.append(d)
                    self.A.append([A1, A2])

                self.e.append(e)
                print('encoder %s'%(layers+1))
                print('decoder %s' % (layers + 1))



        self.pred = tf.layers.dense(self.d[-1], units=self.label_dic_size, activation=tf.nn.relu)
    '''
    def build(self):
        self.x, self.y = self.input_process()
        self.enc_inp, self.A1, self.A2 = self.input_embedding(self.x, n=32, reduce_num=205)

        self.tar_inp, self.tar_real, self.dec_inp = self.decoder_processes(self.y)
        self.e = []


        for layers in range(self.num_layer):
            try:
                e = self.EncoderLayer(self.e[layers-1], head_num=self.head_num, attention_size=self.attention_size)


                self.e.append(e)
            except:
                e = self.EncoderLayer(self.enc_inp, head_num=self.head_num, attention_size=self.attention_size)

                self.e.append(e)

        self.eout = tf.transpose(self.e[-1],[0,2,1])
        self.meout ,self.A_out = self.Muti_attention_layer(self.eout,self.eout,self.eout,attention_size=self.label_length-1
                                                           ,head_num=self.head_num)
        self.meout = tf.transpose(self.meout,[0,2,1])
        self.meout = tf.reshape(self.meout, [-1, self.label_length-1, self.attention_size])
        self.pred = tf.layers.dense(self.meout, units=self.label_dic_size, activation=tf.nn.relu)

    def shuffle_batch(self ,X, y, batch_size):
        rnd_index = np.random.permutation(len(X))
        n_batch = len(X) // batch_size

        for batch_index in np.array_split(rnd_index, n_batch):
            X_batch, y_batch= X[batch_index, :], y[batch_index]
            yield X_batch, y_batch

    def train(self,epoch,batch_size):
        self.batch_size = batch_size
        time_start = datetime.now()
        for epochs in range(epoch):
            time_start_epoch_train = datetime.now()
            batch = self.shuffle_batch(self.input, self.label, batch_size=batch_size)
            mean_cost = 0
            n = 0
            a1 = 0
            a2 = 0
            a3 = 0
            a4 = 0
            for batch_X, batch_y in batch:
                feed = {'input_text:0': batch_X, 'label_text:0': batch_y ,'is_train:0': True}
                _,_, cost,aa1,aa2,aa3,y = self.sess.run([self.add_global,"train_op", "loss:0",
                                                               self.word_accuracy,self.length_accuracy,
                                                                self.WER ,self.y_hat], feed_dict=feed)

                a4 += np.sum(y[:,1] == batch_y[:,1])/batch_size
                mean_cost += cost
                n += 1
                a1 += aa1
                a2 += aa2
                a3 += aa3

            print('-'*50)
            print("Epoch %s"%(epochs + 1))
            mean_cost = mean_cost / n
            print("Train cost : %5f" % (mean_cost))
            self.train_cost.append(mean_cost)
            acc1 = a1/n
            acc2 = a2/n
            acc3 = a3/n
            acc4 = a4/n

            print("Train word acc : %5f" % (acc1))
            print('Train word length acc : %5f'%(acc2))
            print('Train WER : %5f' % (acc3))
            print('First word acc: %5f'%(acc4))
            print(' ')
            self.train_word_acc.append(acc1)
            self.train_length_acc.append(acc2)
            self.train_WER.append(acc3)


            batch = self.shuffle_batch(self.test_input, self.test_label, batch_size=batch_size)
            n = 0
            a1 = 0
            a2 = 0
            a3 = 0
            for batch_X, batch_y in batch:
                pred ,aa1,aa2,aa3 = self.evaluate(batch_X,batch_y = batch_y,acc = True)
                n += 1
                a1 += aa1
                a2 += aa2
                a3 += aa3


            acc1 = a1/n
            acc2 = a2/n
            acc3 = a3/n
            print("Test word acc : %5f" % (acc1))
            print("Test length acc : %5f" % (acc2))
            print("Test WER : %5f" % (acc3))
            self.test_word_acc.append(acc1)
            self.test_length_acc.append(acc2)
            self.test_WER.append(acc3)
            print('During time :%s'%(datetime.now()-time_start_epoch_train))
        print()
        print('Total during time :%s'%(datetime.now()-time_start))

    def evaluate(self ,inp_sentence ,batch_y = None,acc = False):
        inp_sentence = inp_sentence.reshape(-1 ,self.input_length,self.features_num)
        dec_input = np.ones([inp_sentence.shape[0],self.label_length])
        for i in range(self.label_length-1):
            output = self._eval(self.y_hat,
                                feed={'input_text:0': inp_sentence, 'label_text:0': dec_input,'is_train:0': False})
            output = output[:, i]
            if i == (self.label_length - 2):
                if acc == True:
                    word_acc ,length_acc ,WER = self._eval([self.wa,self.la ,self.we] ,
                                                           feed = {'input_text:0': inp_sentence, 'label_text:0': dec_input,
                                                                   'label_text2:0': batch_y[:,1:], 'is_train:0': False})
                    dec_input[:, i + 1] = output
                    return dec_input ,word_acc ,length_acc ,WER
                else:
                    dec_input[:, i + 1] = output
                    return  dec_input
            else:
                dec_input[:, i + 1] = output

    def save(self , name = None, epochs = None,path='./model/'):
        try:
            print('Saving model in %s,with name : %s' % (path, name))
            self.saver.save(self.sess, os.path.join(path, 'model%s.ckpt' % (name)), global_step=epochs)
        except:
            os.makedirs(path)
            print('Saving model in %s,with name : %s' % (path, name))
            self.saver.save(self.sess, os.path.join(path, 'model%s.ckpt' % (name)), global_step=epochs)

    def load(self , path = './model/' ,name = None,):
        print('Loading model from %s' % path)
        self.saver.restore(self.sess, os.path.join(path, 'model%s.ckpt' % (name)))

    def _eval(self , run_state, feed):
        return self.sess.run([run_state], feed_dict=feed)[0]

wav.shape
model = speech_transformer(_input = wav[:10000] ,input_length = 1635  ,label = id_train[:10000],label_length = 50 ,
                           label_dictionary = D_word,test_input = wav[10000:] ,test_label = id_train[10000:],end_num = 28)

model.train(50 ,32)

ii = np.zeros([1,26])
ii[:,0] = 1
ii[:,1] = 5
ii[:,3] = 1855
model._eval(model.y_hat,feed={'input_text:0': [wav[200]], 'label_text:0': [id_train[200]], 'is_train:0': False})
model._eval(model.A_out,feed={'input_text:0': [wav[150]], 'label_text:0': [id_train[150]], 'is_train:0': False})

t = 145
y_pred = model.evaluate(train_mfcc[t])
y_pred = model._eval(model.y_hat,feed={'input_text:0': [train_mfcc[t]], 'label_text:0': [id_train[t]], 'is_train:0': False})
answer = [model.label_dictionary[int(k)] for k in y_pred[0]]
ltt = train_label[t]
print(ltt,'\n',answer)

t = 144
y_pred1 = model.evaluate(train_mel_1[t])
y_pred2 = model.evaluate(train_mel_2[t])
y_pred1 == y_pred2

inp_sentence = train_mel_1[t-1:t]
dec_input = id_train[t-1:t]
feed = {'input_text:0': inp_sentence, 'label_text:0': dec_input, 'is_train:0': False}
a = model._eval(model.pred ,feed)

inp_sentence = train_mfcc[t:t+1]
dec_input = id_train[t:t+1]
feed = {'input_text:0': inp_sentence, 'label_text:0': dec_input, 'is_train:0': False}
b = model._eval(model.pred ,feed)

np.sum(np.abs(a - b))
a.shape


plt.plot(model.train_WER ,label = 'WER')
plt.plot(model.train_word_acc ,label = 'Word acc')
plt.plot(model.train_length_acc ,label = 'Length acc')
plt.plot(model.train_cost ,label = 'train cost')
plt.legend()
plt.show()


inp_sentence.shape
plt.plot(train_mfcc[0][0])
inp_sentence[0][:250]
np.sum(inp_sentence[0][:250]>1)
plt.show()
model._eval(model.enc_inp_11 ,feed).shape

plt.matshow(model._eval(model.A_out ,feed)[0,1])
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
plt.show()
model._eval(model.d ,feed)[-1].shape

t = 5000
inp_sentence = train_mfcc[t:t+1]
dec_input = id_train[t:t+1]
feed = {'input_text:0': inp_sentence, 'label_text:0': dec_input, 'is_train:0': False}
plt.matshow(model._eval(model.A ,feed)[-1][1].reshape(25*4,250))
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
plt.plot(inp_sentence[0,:250,1])
plt.show()

plt.plot(inp_sentence[0,:250])
plt.show()

plt.plot(train_mel_1[100])
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
plt.show()

plt.matshow(np.transpose(train_mfcc[0] ,[1,0]))
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
plt.show()


class speech_transformer_2dA(object):
    def __init__(self, _input, input_length, label, label_length, test_input, test_label, end_num, n_class,
                 dropout_rate=0.1
                 ):
        self.g = tf.Graph()
        self.input_length = input_length
        self.label_length = label_length
        self.features_num = _input.shape[2]
        self.input = _input
        self.label = label
        self.label_dic_size = n_class
        self.dropout_rate = dropout_rate
        self.end_num = end_num

        self.test_input = test_input
        self.test_label = test_label

        self.train_cost = []
        self.train_word_acc = []
        self.train_length_acc = []
        self.test_word_acc = []
        self.test_length_acc = []

        with self.g.as_default():
            ## build the network:
            self.paramater()
            self.build()
            self.y_hat = self.predict()
            # self.y_hat = tf.cast(tf.argmax(self.pred, axis=2),tf.int32)
            self.word_accuracy, self.length_accuracy = self.accuracy_eval(self.y_hat, self.tar_real)
            self.loss = self.loss_function(self.tar_real, self.pred)
            self.optimizer_(self.loss)
            ## initializer
            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        ## create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config, graph=self.g)
        self.sess.run(self.init_op)
        ##writer = tf.summary.FileWriter("TensorBoard/CNN/4Layers", graph = self.sess.graph)

    def accuracy_eval(self, sen_input, target):
        mask = tf.cast(tf.not_equal(0, target), tf.float32)
        word_accuracy = tf.reduce_sum(
            tf.cast(tf.equal(sen_input, target), tf.float32) * mask) / tf.reduce_sum(mask)
        len_acc = tf.equal(tf.reduce_sum(tf.cast(tf.not_equal(0, target), tf.float32), axis=1),
                           tf.reduce_sum(tf.cast(tf.not_equal(0, sen_input), tf.float32), axis=1))
        length_accuracy = tf.reduce_mean(tf.cast(len_acc, tf.float32))

        return word_accuracy, length_accuracy

    def paramater(self):
        self.attention_size = 100
        self.head_num = 3
        self.num_layer = 8
        self.weight = {
            'embedding_wieght': tf.Variable(tf.random_uniform([self.label_dic_size, self.attention_size], -1.0, 1.0),
                                            name='embedding_weight'),
            }

    def input_process(self):
        x = tf.placeholder(tf.float32, [None, self.input_length, self.features_num], name='input_text')
        y = tf.placeholder(tf.int32, [None, self.label_length], name='label_text')

        self.is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
        return x, y

    def get_angles(slef, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        # apply sin to even indices in the array; 2i
        sines = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        cosines = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, cosines], axis=-1)

        pos_encoding = pos_encoding[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def embedding_layer(self, input, name=None):
        embedding = tf.nn.embedding_lookup(self.weight['embedding_wieght'], input, name=name)
        return embedding

    def mask(self, input, mask_tail=None):
        if mask_tail == None:
            mask_seq = tf.cast(tf.equal(input, 0), tf.float32)
        else:
            mask_seq = tf.cast(tf.sequence_mask(mask_tail, self.input_length), tf.float32)
        return mask_seq[:, tf.newaxis, tf.newaxis, :]

    def look_ahead_mask(self, size, name=None):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def self_attention(self, Q, K, V, mask=None):

        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(dk)
        if mask != None:
            A += (mask * -1e9)  # mask 使經過softmax後的padding部分會趨於0
        A = tf.nn.softmax(A)  ##[-1,head_num, q_len, k_len]
        O = tf.matmul(A, V)  ##[batch ,head_num ,sentance_length ,attention_size]

        return O, A

    def head_concate(self, O, attention_size, head_num):
        O = tf.transpose(O, [0, 2, 3, 1])  ##[batch ,sentance_length ,attention_size ,head_num ]
        O_H = tf.reshape(O, [-1, tf.shape(O)[1], attention_size * head_num])
        O_H = tf.layers.dense(O_H, units=attention_size, activation=None)
        return O_H

    def Muti_attention_layer(self, Q, K, V, attention_size, head_num, mask=None):

        ###多維矩陣乘法會將前面維度視為batch，由最後兩維度做矩陣相乘
        Q = tf.layers.dense(Q, units=attention_size * head_num, activation=None)
        Q = tf.reshape(Q, [-1, tf.shape(Q)[1], head_num, attention_size])
        Q = tf.transpose(Q, [0, 2, 1, 3])  ##[-1,head_num, q_len, attention_size ]
        K = tf.layers.dense(K, units=attention_size * head_num, activation=None)
        K = tf.reshape(K, [-1, tf.shape(K)[1], head_num, attention_size])
        K = tf.transpose(K, [0, 2, 1, 3])  ##[-1,head_num, k_len , attention_size ]
        V = tf.layers.dense(V, units=attention_size * head_num, activation=None)
        V = tf.reshape(V, [-1, tf.shape(V)[1], head_num, attention_size])
        V = tf.transpose(V, [0, 2, 1, 3])  ##[-1,head_num, v_len, attention_size ]

        O, A = self.self_attention(Q, K, V, mask=mask)
        O_H = self.head_concate(O, attention_size, head_num)

        ##[batch ,head_num ,sentance_length ,attention_size]

        return O_H, A

    def TwoD_attention(self, input, n, c):

        '''
        1.attention alone time
        '''
        conv_Q_filter = tf.Variable(tf.random_normal([3, 3, n, c], stddev=0.01))
        conv_K_filter = tf.Variable(tf.random_normal([3, 3, n, c], stddev=0.01))
        conv_V_filter = tf.Variable(tf.random_normal([3, 3, n, c], stddev=0.01))
        conv_filter = tf.Variable(tf.random_normal([3, 3, n, c], stddev=0.01))
        conv_Q = tf.nn.conv2d(input, conv_Q_filter, strides=[1, 1, 1, 1], padding="SAME")
        conv_Q = tf.transpose(conv_Q, [0, 3, 1, 2])  # [batch ,head num ,window num ,mfcc feature]
        conv_K = tf.nn.conv2d(input, conv_K_filter, strides=[1, 1, 1, 1], padding="SAME")
        conv_K = tf.transpose(conv_K, [0, 3, 1, 2])  # [batch ,head num ,window num ,mfcc feature]
        conv_V = tf.nn.conv2d(input, conv_V_filter, strides=[1, 1, 1, 1], padding="SAME")
        conv_V = tf.transpose(conv_V, [0, 3, 1, 2])  # [batch ,head num  ,window num ,mfcc feature]
        conv_inp = tf.nn.conv2d(input, conv_filter, strides=[1, 1, 1, 1], padding="SAME")
        conv_inp = tf.transpose(conv_inp, [0, 3, 1, 2])  # [batch ,head num  ,window num ,mfcc feature]
        O, A = self.self_attention(conv_Q, conv_K, conv_V, mask=None)
        out = tf.contrib.layers.layer_norm(O + conv_inp)
        out = tf.transpose(out, [0, 2, 3, 1])

        '''
        concat convalution output
        '''
        out1_filter = tf.Variable(tf.random_normal([3, 3, c, c], stddev=0.01))
        out2_filter = tf.Variable(tf.random_normal([3, 3, c, c], stddev=0.01))
        out1 = tf.nn.conv2d(out, out1_filter, strides=[1, 1, 1, 1], padding='SAME')
        out1 = tf.layers.batch_normalization(out1)
        out2 = tf.nn.conv2d(out1, out2_filter, strides=[1, 1, 1, 1], padding='SAME')
        out2 = tf.layers.batch_normalization(out2)
        output = tf.nn.relu(out2 + out)
        ## [batch ,window num ,mfcc feature,n]

        return output, A

    def feed_foward_network(self, input, hidden_units):
        size = input.shape[-1]
        h1 = tf.layers.dense(input, units=hidden_units, activation=tf.nn.relu)
        out = tf.layers.dense(h1, units=size, activation=None)
        return out

    def input_embedding(self, _input, n, reduce_num):
        _input = _input[:, :, :, tf.newaxis]
        conv1_filter = tf.Variable(tf.random_normal([3, 3, 1, n], stddev=0.01))
        conv1_enc_inp = tf.nn.conv2d(_input, conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv1_enc_inp = tf.layers.batch_normalization(conv1_enc_inp)
        conv2_filter = tf.Variable(tf.random_normal([3, 3, n, n], stddev=0.01))
        conv2_enc_inp = tf.nn.conv2d(conv1_enc_inp, conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv2_enc_inp = tf.layers.batch_normalization(conv2_enc_inp)

        enc_inp_1_twod, A1 = self.TwoD_attention(conv2_enc_inp, n=n, c=n)
        enc_inp_2_twod, A2 = self.TwoD_attention(enc_inp_1_twod, n=n, c=n)
        out = tf.reshape(enc_inp_2_twod, [-1, reduce_num, self.features_num * n])
        enc_inp_out = tf.layers.dense(out, units=self.attention_size)
        enc_inp_out = tf.contrib.layers.layer_norm(enc_inp_out)

        enc_inp_out += self.positional_encoding(reduce_num, self.attention_size)
        enc_inp_out = tf.layers.dropout(enc_inp_out, rate=self.dropout_rate, training=self.is_train)
        return enc_inp_out, A1, A2

    def decoder_processes(self, _input):
        tar_inp = _input[:, :-1]
        tar_real = _input[:, 1:]

        dec_inp = self.embedding_layer(tar_inp)
        dec_inp += self.positional_encoding(self.label_length - 1, self.attention_size)
        return tar_inp, tar_real, dec_inp

    def EncoderLayer(self, input, head_num, attention_size):

        O, A = self.Muti_attention_layer(input, input, input, attention_size=attention_size, head_num=head_num)
        O = tf.layers.dropout(O, rate=self.dropout_rate, training=self.is_train)
        out1 = tf.contrib.layers.layer_norm(O + input)

        ffout = self.feed_foward_network(out1, attention_size * 4)
        ffout = tf.layers.dropout(ffout, rate=self.dropout_rate, training=self.is_train)
        out2 = tf.contrib.layers.layer_norm(ffout + out1)


        return out2

    def DecoderLayer(self, input, enc_input, head_num, attention_size, inp_mask=None):
        '''
        O, A1 = self.Muti_attention_layer(input, input, input, attention_size=attention_size, head_num=head_num,
                                          mask=combined_mask)
        O = tf.layers.dropout(O, rate=self.dropout_rate, training=self.is_train)
        ##out1 = tf.contrib.layers.layer_norm(O+input)
        out1 = O + input
        '''
        O_1, A2 = self.Muti_attention_layer(Q=input, K=enc_input, V=enc_input, attention_size=attention_size,
                                            head_num=head_num, mask=inp_mask)
        ##這裡的mask遮掉的是encoder的padding部分,因此使用encoder 的mask
        O_1 = tf.layers.dropout(O_1, rate=self.dropout_rate, training=self.is_train)
        # out2 = tf.contrib.layers.layer_norm(O_1+out1)
        out2 = O_1 + input

        ffout = self.feed_foward_network(out2, attention_size * 4)
        ffout = tf.layers.dropout(ffout, rate=self.dropout_rate, training=self.is_train)
        # out3 = tf.contrib.layers.layer_norm(ffout + out2)
        out3 = ffout + out2

        return out3, A2

    def loss_function(self, real, pred):
        # loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred)
        # mask = tf.logical_not(tf.equal(real, 0))
        # mask = tf.cast(mask, dtype=loss_.dtype)
        # loss_ *= mask  # 只計算非 <pad> 位置的損失

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        loss_ = loss_object(real, pred)
        self.l = loss_
        mask = tf.logical_not(tf.equal(real, 0))
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask  # 只計算非 <pad> 位置的損失

        return tf.reduce_mean(loss_, name='loss')

    def optimizer_(self, loss):
        global_step = tf.Variable(0, dtype=tf.float32, trainable=False)
        warmup_steps = 4000
        arg1 = tf.rsqrt(global_step)
        arg2 = global_step * (warmup_steps ** -1.5)
        learning_rate = tf.rsqrt(float(100)) * tf.minimum(arg1, arg2)
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98,
                                     epsilon=1e-9).minimize(loss, name='train_op')
        self.add_global = global_step.assign_add(1)

    def predict(self):
        y_hat = tf.cast(tf.argmax(self.pred, axis=2), tf.int32)
        c = tf.ones_like(y_hat)[:, 0] * self.end_num
        c = tf.expand_dims(c, 1)
        y_hat_ex = tf.concat([y_hat, c], axis=1)
        ind = tf.where(tf.equal(y_hat_ex, self.end_num))
        ind = tf.segment_min(ind[:, 1], ind[:, 0]) + 1
        mask = tf.cast(tf.sequence_mask(ind, self.label_length), tf.int32)
        result = (y_hat_ex * mask)[:, :-1]

        return result

    def build(self):
        self.x, self.y = self.input_process()
        self.enc_inp, self.A1, self.A2 = self.input_embedding(self.x, n=64, reduce_num=50)

        self.tar_inp, self.tar_real, self.dec_inp = self.decoder_processes(self.y)
        m = self.mask(self.tar_inp)
        self.e = []
        self.d = []
        self.A = []

        for layers in range(self.num_layer):
            try:
                e = self.EncoderLayer(self.e[layers-1], head_num=self.head_num, attention_size=self.attention_size)

                d,  A2 = self.DecoderLayer(self.d[layers-1], e, head_num=self.head_num,
                                              attention_size=self.attention_size)
                self.d.append(d)
                self.A.append(A2)

                self.e.append(e)
            except:
                e = self.EncoderLayer(self.enc_inp, head_num=self.head_num, attention_size=self.attention_size)

                d, A2 = self.DecoderLayer(self.dec_inp, e, head_num=self.head_num,
                                              attention_size=self.attention_size)
                self.d.append(d)
                self.A.append(A2)
                self.e.append(e)

        self.pred = tf.layers.dense(self.d[-1], units=self.label_dic_size, activation=tf.nn.relu)

    def shuffle_batch(self, X, y, batch_size):
        rnd_index = np.random.permutation(len(X))
        n_batch = len(X) // batch_size

        for batch_index in np.array_split(rnd_index, n_batch):
            X_batch, y_batch = X[batch_index, :], y[batch_index]
            yield X_batch, y_batch

    def train(self, epoch, batch_size):
        self.batch_size = batch_size
        time_start = datetime.now()
        for epochs in range(epoch):
            time_start_epoch_train = datetime.now()
            batch = self.shuffle_batch(self.input, self.label, batch_size=batch_size)
            mean_cost = 0
            n = 0
            a1 = 0
            a2 = 0
            for batch_X, batch_y in batch:
                feed = {'input_text:0': batch_X, 'label_text:0': batch_y, 'is_train:0': True}
                _, _, cost, aa1, aa2 = self.sess.run(
                    [self.add_global, "train_op", "loss:0", self.word_accuracy, self.length_accuracy], feed_dict=feed)

                mean_cost += cost
                n += 1
                a1 += aa1
                a2 += aa2

            print('-' * 50)
            print("Epoch %s" % (epochs + 1))
            mean_cost = mean_cost / n
            print("Train cost : %5f" % (mean_cost))
            self.train_cost.append(mean_cost)
            acc1 = a1 / n
            acc2 = a2 / n

            print("Train word acc : %5f" % (acc1))
            print('Train word length acc : %5f' % (acc2))
            print(' ')
            self.train_word_acc.append(acc1)
            self.train_length_acc.append(acc2)

            batch = self.shuffle_batch(self.test_input, self.test_label, batch_size=batch_size)
            n = 0
            a1 = 0
            a2 = 0
            for batch_X, batch_y in batch:
                pred = self.evaluate(batch_X)
                aa1 = np.sum(pred[:, 1] == batch_y[:, 1]) / batch_size
                #aa2 = np.sum(pred[:, 2] == batch_y[:, 2]) / batch_size
                n += 1
                a1 += aa1
                #a2 += aa2

            acc1 = a1 / n
            #acc2 = a2 / n
            print("Test word acc : %5f" % (acc1))
            #print("Test length acc : %5f" % (acc2))
            self.test_word_acc.append(acc1)
            #self.test_length_acc.append(acc2)
            print('During time :%s' % (datetime.now() - time_start_epoch_train))
        print()
        print('Total during time :%s' % (datetime.now() - time_start))
        plt.plot(self.train_word_acc, label='Train Word acc')
        plt.plot(self.test_word_acc, label='Test Word acc')
        plt.legend()
        plt.title('Acc')
        plt.show()
        print(max(self.test_word_acc))

    def evaluate(self, inp_sentence):
        inp_sentence = inp_sentence.reshape(-1, self.input_length, self.features_num)
        dec_input = np.zeros([inp_sentence.shape[0], self.label_length])
        dec_input[:, 0] = dec_input[:, 0] + 1392  ##<BOS>
        for i in range(self.label_length - 1):
            output = self._eval(self.y_hat,
                                feed={'input_text:0': inp_sentence, 'label_text:0': dec_input, 'is_train:0': False})
            output = output[:, i]
            dec_input[:, i + 1] = dec_input[:, i + 1] + output
            if output.shape[0] == 1:
                if output == self.end_num:
                    return dec_input
        return dec_input

    def save(self, name=None, epochs=None, path='./model/'):
        try:
            print('Saving model in %s,with name : %s' % (path, name))
            self.saver.save(self.sess, os.path.join(path, 'model%s.ckpt' % (name)), global_step=epochs)
        except:
            os.makedirs(path)
            print('Saving model in %s,with name : %s' % (path, name))
            self.saver.save(self.sess, os.path.join(path, 'model%s.ckpt' % (name)), global_step=epochs)

    def load(self, path='./model/', name=None, ):
        print('Loading model from %s' % path)
        self.saver.restore(self.sess, os.path.join(path, 'model%s.ckpt' % (name)))

    def _eval(self, run_state, feed=None):
        if feed == None:
            return self.sess.run([run_state], feed_dict={'input_text:0': self.input, 'label_text:0': self.label,
                                                         'is_train:0': False})[0]
        else:
            return self.sess.run([run_state], feed_dict=feed)[0]


model_basic_2dA = speech_transformer_2dA(_input=train_mfcc[:,:50], input_length=train_mfcc[:,:50].shape[1], label=id_train[:,[0,1,-1]],
                                 label_length=3,
                                 test_input=test_mfcc[:,:50], test_label=id_test[:,[0,1,-1]], end_num=8, n_class=1981,
                                 dropout_rate=0.1
                                 )

model_basic_2dA .train(5,128)

model_basic_2dA._eval(model_basic_2dA.y_hat,feed={'input_text:0': train_mfcc[200:205][:,:50], 'label_text:0': id_train[200:205][:,[0,1,-1]], 'is_train:0': False})


