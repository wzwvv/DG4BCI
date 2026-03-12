import random

from torch.utils.data import Dataset
import torch
import scipy.io
import numpy as np
from scipy import signal
from scipy.io import loadmat
import os
from etc.global_config import config
import itertools
from Utils.dataprocess import data_aug

class getSSVEPIntra(Dataset):
   def __init__(self, subject=1, train_ratio=0.8,kfold = 0,info=[]):
       super(getSSVEPIntra, self).__init__()
       self.datasetid = config["train_param"]["datasets"]
       self.Nh = config[f"data_param{self.datasetid}"]["Nh"]  # number of trials
       self.Nc = config[f"data_param{self.datasetid}"]["Nc"] # number of channels
       self.Nt = config[f"data_param{self.datasetid}"]["Nt"]  # number of time points
       self.Nf = config[f"data_param{self.datasetid}"]["Nf"]  # number of target frequency
       self.Fs = config[f"data_param{self.datasetid}"]["Fs"] # Sample Frequency
       self.ws = config["train_param"]["ws"]   # window size of ssvep
       self.Nm = config["model_param"]["Nm"]
       self.info = info
       augtype=config['train_param']['augtype']
       self.T = int(self.Fs * self.ws)
       self.subject = subject  # current subject
       data_paths = config.get('data_path', {})
       if self.datasetid==1:
           dp = data_paths.get('benchmark12', r"D:\Project\gra_design\Dataset\benchmark12")
           self.data = self.load_1Data(data_path=dp,segment=self.ws)
           self.indexid =self.intra_benchmark12_split()
       elif self.datasetid == 2:
           dp = data_paths.get('benchmark40', r"D:\Project\gra_design\Dataset\benchmark40")
           self.data = self.load_2Data(dp,segment=self.ws)
           self.indexid =self.intra_benchmark40_split(train_ratio)
       elif self.datasetid == 3:
           dp = data_paths.get('BETA', '/data2/hzt/ssvep/BETA')
           self.data = self.load_3Data(dp,segment=self.ws)
           self.indexid =self.intra_benchmark40_split(train_ratio)
       self.eeg_data=self.data[0]
       self.label_data=self.data[1]

       self.aug_data,self.aug_label=data_aug(self.eeg_data,self.label_data,augtype=augtype,info=self.info)
       self.eeg_data=self.filter_bank(self.eeg_data) #(self.Nh, max(self.Nm,1), self.Nc, self.T))
       self.aug_data=self.filter_bank(self.aug_data)
       
       self.train_idx = []
       self.aug_train_idx = []
       self.test_idx = []
       for i in range(0, self.Nh, self.Nh // self.Nf):
           for j in range(self.Nh // self.Nf):
               if train_ratio < 0.5 :
                if j in self.indexid[kfold]:
                    self.train_idx.append(i + j)
                    self.aug_train_idx.append(i + j)
                    self.aug_train_idx.append(i + j + self.Nh)
                else:
                    self.test_idx.append(i + j)
               else:
                   if j in self.indexid[kfold]:
                    self.test_idx.append(i + j)
                   else:
                    self.train_idx.append(i + j)
                    self.aug_train_idx.append(i + j)
                    self.aug_train_idx.append(i + j + self.Nh)
                    
                

       self.eeg_data_train = self.eeg_data[self.train_idx]
       self.aug_eeg_train = self.aug_data[self.aug_train_idx]
       
       self.label_data_train = self.label_data[self.train_idx]
       self.aug_label_train = self.aug_label[self.aug_train_idx]
       #print(self.label_data_train,self.aug_label_train)
       
       self.eeg_data_test = self.eeg_data[self.test_idx]
       self.label_data_test = self.label_data[self.test_idx]
       # torch.save(self.eeg_data_train,'eeg_data_train.pt')
       # torch.save(self.label_data_train, 'label_data_train.pt')
       # torch.save(self.eeg_data_test, 'eeg_data_test.pt')
       # torch.save(self.label_data_test,'label_data_test.pt')

   def __getitem__(self,index):
       return (self.eeg_data_train, self.aug_eeg_train,self.label_data_train,self.aug_label_train),(self.eeg_data_test,self.label_data_test)

   def __len__(self):
       return len(self.label_data)

   def filter_bank(self,eeg):
       result = np.zeros((eeg.shape[0], max(self.Nm,1), self.Nc, self.T))

       nyq = self.Fs / 2
       
       if self.datasetid == 1:
           passband = [9, 18, 27, 36, 45, 54]
           stopband = [7, 15, 19, 28, 37, 46]

       else:
           passband = [8, 16, 24, 32, 40, 48]
           stopband = [6, 12, 18, 26, 34, 42]
    #    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    #    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]

       highcut_pass, highcut_stop = 80,90

       gpass, gstop, Rp = 3, 40, 0.5
       if self.Nm:
           for i in range(self.Nm):
               Wp = [passband[i] / nyq, highcut_pass / nyq]
               Ws = [stopband[i] / nyq, highcut_stop / nyq]
               [N, Wn] = signal.cheb1ord(Wp, Ws, gpass, gstop)
               [B, A] = signal.cheby1(N, Rp, Wn, 'bandpass')

               data = signal.filtfilt(B, A, eeg, padlen=3 * (max(len(B), len(A)) - 1)).copy()
               result[:, i, :, :] = data
       else:
           #return eeg
           result[:, 0, :, :]=eeg
       return result 

   def load_1Data(self, data_path='Dataset/benchmark12',decay=0.14, segment=1):
        subjectfile = loadmat(os.path.join(data_path, f'S{self.subject}.mat'))
        samples = subjectfile['eeg']  # (12, 8, 1024, 15)
        eeg_data = samples[0, :, :, :]  # (8, 1024, 15)
        label_data = np.zeros((180, 1))
        for i in range(1, 12):
            eeg_data = np.concatenate([eeg_data, samples[i, :, :, :]], axis=2)
            label_data[i * 15:(i + 1) * 15] = i
        eeg_data = eeg_data.transpose([2, 0, 1])  # (8, 1114, 180) -> (180, 8, 1024)
        eeg_data = eeg_data[:, :, int((decay) * 256):int(( decay + segment) * 256)]
        return eeg_data,label_data #(180,8,256)

   def load_2Data(self, data_path='Dataset/benchmark40', decay=0.14, segment=1):
       data = loadmat(os.path.join(data_path, f'S{self.subject}.mat'))
       x0 = data['data'][[47, 53, 54, 55, 56, 57, 60, 61, 62], int((0.5 + decay) * 250):int((0.5 + decay + segment) * 250),:,:]#(9, 250, 40, 6)
       x0 = x0.transpose(2,0,1,3) #(40, 9, 250, 6)
       eeg_data = x0[0] 
       label_data = np.zeros((240, 1))# )
       for i in range(1, self.Nf):
            eeg_data = np.concatenate([eeg_data, x0[i]],axis=2)
            label_data[i * 6:(i + 1) * 6] = i
       eeg_data = eeg_data.transpose([2, 0, 1]) #(240,9,250)
       return eeg_data, label_data 
   
   def intra_benchmark40_split(self,ratio=0.85):
    num_blocks = 6
    num_groups = 1
    blocks_per_group = num_blocks // num_groups  # 6

    num_samples = 6 - int(6 * ratio)  # 每次选择 1 个
    # 三组 block 索引：[[0-5], [6-11], [12-17]]
    groups = [list(range(i * blocks_per_group, (i + 1) * blocks_per_group)) for i in range(num_groups)]

    combinations_per_group = [list(itertools.combinations(g, num_samples)) for g in groups]
    # 检查各组组合数量是否相等，否则取最小长度对齐
    min_len = min(len(c) for c in combinations_per_group)

    # 合并：对应索引的三组组合打平
    # merged = [
    # combinations_per_group[0][i] + combinations_per_group[1][i] + combinations_per_group[2][i]
    # for i in range(min_len)
    # ]
    merged = [
        combinations_per_group[0][i]
        for i in range(min_len)
    ]
    return merged
   def intra_benchmark12_split(self,num_splits=5):
        splits=[]
        for i in range(num_splits):
            train_subjects = [ i*3+j for j in range(3)]

            splits.append(train_subjects)
        return splits

class getSSVEPInter(Dataset):
   def __init__(self, trainsubject=[1], testsubject=[2]):
       super(getSSVEPInter, self).__init__()
       self.datasetid = config["train_param"]["datasets"]
       self.Nh = config[f"data_param{self.datasetid}"]["Nh"]  # number of trials
       self.Nc = config[f"data_param{self.datasetid}"]["Nc"]  # number of channels
       self.Nt = config[f"data_param{self.datasetid}"]["Nt"]  # number of time points
       self.Nf = config[f"data_param{self.datasetid}"]["Nf"]  # number of target frequency
       self.Fs = config[f"data_param{self.datasetid}"]["Fs"]  # Sample Frequency
       self.ws = config["train_param"]["ws"]  # window size of ssvep
       self.Nm = config["model_param"]["Nm"]
       self.T = int(self.Fs * self.ws)
       self.trainsubject = trainsubject  # current subject
       self.testsubject = testsubject
       for subject in self.trainsubject:
           data = []
           if self.datasetid == 1:
               data = self.load_1Data('../../data2/hzt/ssvep/benchmark12',subject,decay=0.14, segment=1)
           elif self.datasetid == 2:
               data = self.load_2Data('../../data2/hzt/ssvep/benchmark40',subject,decay=0.14, segment=1)
           eeg_data = data[0]
           label_data = data[1].reshape(self.Nf,-1)
           eeg_data = self.filter_bank(eeg_data).reshape(self.Nf,-1,max(self.Nm,1), self.Nc, self.T)  #(self.Nh, max(self.Nm,1), self.Nc, self.T)
           if not hasattr(self, 'eeg_data_train'):
               self.eeg_data_train = eeg_data
               self.label_data_train = label_data
           else:
               self.eeg_data_train = np.concatenate([self.eeg_data_train, eeg_data], axis=1)
               self.label_data_train = np.concatenate([self.label_data_train, label_data], axis=1)
       self.eeg_data_train = self.eeg_data_train.reshape(-1,max(self.Nm,1), self.Nc, self.T)
       self.label_data_train =self.label_data_train.reshape(-1,1)
       for subject in self.testsubject:
           data = []
           if self.datasetid == 1:
               data = self.load_1Data('../../data2/hzt/ssvep/benchmark12',subject,decay=0.14, segment=1)
           elif self.datasetid == 2:
               data = self.load_2Data('../../data2/hzt/ssvep/benchmark40',subject,decay=0.14, segment=1)
           eeg_data = data[0]
           label_data = data[1].reshape(self.Nf,-1)
           eeg_data = self.filter_bank(eeg_data).reshape(self.Nf,-1,max(self.Nm,1), self.Nc, self.T)  # 第一次赋值  #(self.Nh, max(self.Nm,1), self.Nc, self.T)
           if not hasattr(self, 'eeg_data_test'):
               self.eeg_data_test = eeg_data
               self.label_data_test = label_data
           else:
               self.eeg_data_test = np.concatenate([self.eeg_data_test, eeg_data], axis=1)
               self.label_data_test = np.concatenate([self.label_data_test, label_data], axis=1)
       self.eeg_data_test = self.eeg_data_test.reshape(-1, max(self.Nm, 1), self.Nc, self.T)
       self.label_data_test = self.label_data_test.reshape(-1, 1)

   def __getitem__(self, index):
       return (self.eeg_data_train, self.label_data_train), (self.eeg_data_test, self.label_data_test)

   def __len__(self):
       return len(self.label_data_train),len(self.label_data_test)

   def filter_bank(self, eeg):
       result = np.zeros((self.Nh, max(self.Nm, 1), self.Nc, self.T))

       nyq = self.Fs / 2
       passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
       stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
       if self.datasetid == 2:
           passband = [8, 16, 24, 32, 40, 48]
           stopband = [6, 12, 18, 26, 34, 42]
       else:
           passband = [9, 18, 27, 36, 45, 54]
           stopband = [7, 15, 19, 28, 37, 46]

       highcut_pass, highcut_stop = 64, 70

       gpass, gstop, Rp = 3, 40, 0.5
       if self.Nm:
           for i in range(self.Nm):
               Wp = [passband[i] / nyq, highcut_pass / nyq]
               Ws = [stopband[i] / nyq, highcut_stop / nyq]
               [N, Wn] = signal.cheb1ord(Wp, Ws, gpass, gstop)
               [B, A] = signal.cheby1(N, Rp, Wn, 'bandpass')

               data = signal.filtfilt(B, A, eeg, padlen=3 * (max(len(B), len(A)) - 1)).copy()
               result[:, i, :, :] = data
       else:
           result[:, 0, :, :] = eeg
       return result

   def load_1Data(self, data_path='Dataset/benchmark12', subject = 1 ,decay=0.14, segment=1):
       subjectfile = loadmat(os.path.join(data_path, f'S{subject}.mat'))
       samples = subjectfile['eeg']  # (12, 8, 1024, 15)
       eeg_data = samples[0, :, :, :]  # (8, 1024, 15)
       label_data = np.zeros((180, 1))
       for i in range(1, 12):
           eeg_data = np.concatenate([eeg_data, samples[i, :, :, :]], axis=2)
           label_data[i * 15:(i + 1) * 15] = i
       eeg_data = eeg_data.transpose([2, 0, 1])  # (8, 1114, 180) -> (180, 8, 1024)
       eeg_data = eeg_data[:, :, int((decay) * 256):int((decay + segment) * 256)]
       return eeg_data, label_data

   def load_2Data(self, data_path='Dataset/benchmark40',subject = 1 ,decay=0.14, segment=1):
       data = loadmat(os.path.join(data_path, f'S{subject}.mat'))
       x = data['data'][:, int((0.5 + decay) * 250):int((0.5 + decay + segment) * 250), :, :]  # (64, 250, 40, 6)
       x = x.transpose(2, 0, 1, 3)  # (40, 64, 250, 6)
       eeg_data = x[0, :, :, :]  # (64, 250, 6)
       label_data = np.zeros((240, 1))  # )
       for i in range(1, self.Nf):
           eeg_data = np.concatenate([eeg_data, x[i, :, :, :]], axis=2)
           label_data[i * 6:(i + 1) * 6] = i
       eeg_data = eeg_data.transpose([2, 0, 1])
       return eeg_data, label_data

def cross_subject_split(subjects, num_splits=5):
    splits = []
    # 每次进行一次跨被试分割
    for i in range(num_splits):
        # 根据题意每次选择 (i, i+5) 这样的组合
        test_subjects = [subjects[i+j*num_splits] for j in range(len(subjects)//num_splits)]
        # 训练集是除去这两个测试被试的所有被试
        train_subjects = [subject for subject in subjects if subject not in test_subjects]

        splits.append({
            'train': train_subjects,
            'test': test_subjects
        })
    return splits
def z_score_normalization(eeg_data):
    mean = np.mean(eeg_data, axis=-1, keepdims=True)
    std = np.std(eeg_data, axis=-1, keepdims=True)
    return (eeg_data - mean) / (std + 1e-6)

