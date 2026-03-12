import torch
import torch.nn as nn
import numpy as np
from scipy import signal
import matplotlib.pyplot as  plt
import math
from scipy.linalg import qr
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from scipy import signal
from scipy.linalg import eigh
from scipy.linalg import solve
from scipy.io import loadmat
import os

def isPD(B):
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def nearestPD(A):
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    print("Replace current matrix with the nearest positive-definite matrix.")

    spacing = np.spacing(np.linalg.norm(A))
    eye = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += eye * (-mineig * k ** 2 + spacing)
        k += 1

    return A3
class KLGlayer(nn.Module):
    def __init__(self, kernel_size=33, eeg_data_train=None, label_data_train=None, eeg_data_test=None, label_data_test=None,config = None,output=0):
        super(KLGlayer, self).__init__()
        
        datasetid = config['train_param']['datasets']
        self.deltafreq = config[f'data_param{datasetid}']['deltafreq']
        self.initfreq = config[f'data_param{datasetid}']['initfreq']
        self.Fs = config[f'data_param{datasetid}']['Fs']
        self.Nf = config[f'data_param{datasetid}']['Nf']
        self.Nm = max(1,config['model_param']['Nm'])
        self.Nc= config[f'data_param{datasetid}']['Nc']
        self.Nt = eeg_data_train.shape[-1]
        self.kernel_size=kernel_size
        self.output = output
        self.eeg_data_train = np.transpose(eeg_data_train.reshape(self.Nf, -1, self.Nm, self.Nc, self.Nt),
                                           (0, 2, 3, 4, 1))
        self.eeg_data_test = np.transpose(eeg_data_test.reshape(self.Nf, -1, self.Nm, self.Nc, self.Nt),
                                          (0, 2, 3, 4, 1))
        self.label_data_test = label_data_test.reshape(self.Nf, -1)
        self.KLGalgorithm = config['KLGalgorithm']
        self.Bandpass = config['model_param']['Bandpass']
        self.Bandpass_FIX = config[self.KLGalgorithm]['Bandpass_FIX']
        self.slc_all = config[self.KLGalgorithm]['slc_all']
        self.n_components =config[self.KLGalgorithm]['n_components']
        self.KLG_FIX = config[self.KLGalgorithm]['KLG_FIX']
        self.MIX = config['model_param']['MIX']
        self.DL = config['model_param']['DL']
        self.FB = config['model_param']['FB']
        if datasetid ==2 and not self.slc_all:
                self.channels = self.select_channel()#[47, 53, 54, 55, 56, 57, 60, 61, 62]
        if datasetid ==3 and not self.slc_all:
                self.channels = self.select_channel_BETA()#[47, 53, 54, 55, 56, 57, 60, 61, 62]
        else:
            self.channels =[i for i in range(self.Nc)]
            # 绘制原始信号
        # plt.figure(figsize=(10, 6))
        # plt.subplot(2, 1, 1)
        # 计算信号的频谱（通过FFT）
        # eeg=self.Bandpasslayer(torch.tensor(eeg_data_train, dtype=torch.float32)).detach().numpy()
        # print(eeg.shape)
        # f_signal = np.fft.fft(np.mean(eeg[12, 0, self.channels, :], axis=0))
        #
        # # print(label_data_train[12])
        # f_signal_freq = np.fft.fftfreq(250, 1 / 250)
        # # print(f_signal)
        # f_signal_magnitude = np.abs(f_signal[:125])
        # sorted_indices = np.argsort(f_signal_magnitude)[::-1]
        # plt.plot(f_signal_freq[:125], f_signal_magnitude)
        # plt.title("Original Signal")
        # plt.xlabel("Time [s]")
        # plt.ylabel("Amplitude")
        # plt.subplot(2, 1, 2)
        # plt.plot([i for i in range(250)], np.mean(eeg[0, 0, self.channels, :], axis=0))
        #print(np.mean(eeg[0, 0, self.channels, :], axis=0))
        #plt.show()
        if self.Bandpass:
            self.Bandpasslayer = BandpassConvLayer(Fs=self.Fs, initfreq=self.initfreq, deltafreq=self.deltafreq, Nf=self.Nf,
                                              kernel_size=self.kernel_size,Nm=self.Nm,Bandpass_FIX=self.Bandpass_FIX)
            result1 = self.Bandpasslayer.forward(torch.tensor(eeg_data_train, dtype=torch.float32))

            result2 = np.transpose(
                self.Bandpasslayer.forward(torch.tensor(eeg_data_test, dtype=torch.float32)).detach().numpy().reshape(
                    self.Nf, -1, self.Nm, self.Nc, self.Nt), (0, 2, 3, 4, 1))
            if self.KLGalgorithm=='TRCA':
                self.knowlayer = TRCAlayer(self.KLG_FIX,
                                           np.transpose(result1.reshape(self.Nf, -1, self.Nm, self.Nc, self.Nt).detach().numpy(),
                                                        (0, 2, 3, 4,1)), self.channels,self.DL)
                #self.knowlayer = TRCAlayer(self.KLG_FIX, self.eeg_data_train, self.channels)
                self.knowlayer.trca(result1.detach().numpy(),label_data_train.reshape(-1))
                self.init1_acc = self.knowlayer.test_trca(result2, self.label_data_test)
                self.mixlayer = MIXlayer(mix_FIX=config['MIXlayer']['mix_FIX'], Nf=self.Nf, Nm=self.Nm,
                                         DL=self.DL)
                self.fblayer = FBlayer(FB_FIX=config[self.KLGalgorithm]['FB_FIX'], Nf=self.Nf, Nm=self.Nm, DL=self.DL)
            elif self.KLGalgorithm=='TDCA':

                self.knowlayer = TDCAlayer(self.KLG_FIX,
                                           np.transpose(result1.reshape(self.Nf, -1, self.Nm, self.Nc,
                                                                        self.Nt).detach().numpy(),
                                                        (0, 2, 3, 4, 1)), self.channels, self.DL,n_components=self.n_components)
                # self.knowlayer = TRCAlayer(self.KLG_FIX, self.eeg_data_train, self.channels)
                self.knowlayer.tdca(result1.detach().numpy(), label_data_train.reshape(-1))
                self.init1_acc = self.knowlayer.test_tdca(result2, self.label_data_test)
                self.mixlayer = MIXlayer(mix_FIX=config['MIXlayer']['mix_FIX'], Nf=self.n_components, Nm=self.Nm,
                                         DL=self.DL)
                self.fblayer = FBlayer(FB_FIX=config[self.KLGalgorithm]['FB_FIX'], Nf=self.n_components, Nm=self.Nm,
                                       DL=self.DL)
            else :
                self.know1layer = TDCAlayer(self.KLG_FIX,
                                           np.transpose(result1.reshape(self.Nf, -1, self.Nm, self.Nc,
                                                                        self.Nt).detach().numpy(),
                                                        (0, 2, 3, 4, 1)), self.channels, self.DL,
                                           n_components=self.n_components)
                # self.knowlayer = TRCAlayer(self.KLG_FIX, self.eeg_data_train, self.channels)
                self.know1layer.tdca(result1.detach().numpy(), label_data_train.reshape(-1))
                self.init1_acc = self.know1layer.test_tdca(result2, self.label_data_test)
                self.know2layer = TRCAlayer(self.KLG_FIX, np.transpose(
                    self.know1layer(result1).detach().numpy().reshape(self.Nf, -1, self.Nm,
                                                                                 self.n_components, self.Nt),(0, 2, 3, 4, 1)),
                                            [i for i in range(self.n_components)], self.DL)

                self.know2layer.trca(
                    self.know1layer(result1).detach().numpy(),
                    label_data_train.reshape(-1))

                self.init2_acc = self.know2layer.test_trca(np.transpose(
                    self.know1layer(self.Bandpasslayer.forward(torch.tensor(eeg_data_test, dtype=torch.float32))).detach().numpy().reshape(self.Nf, -1, self.Nm,
                                                                                 self.n_components, self.Nt),
                    (0, 2, 3, 4, 1)), self.label_data_test)


                self.knowlayer = nn.Sequential(self.know1layer,self.know2layer)
                self.mixlayer = MIXlayer(mix_FIX=config['MIXlayer']['mix_FIX'], Nf=self.Nf, Nm=self.Nm,
                                         DL=self.DL)
                self.fblayer = FBlayer(FB_FIX=config[self.KLGalgorithm]['FB_FIX'], Nf=self.Nf, Nm=self.Nm, DL=self.DL)

        else:
            if self.KLGalgorithm == 'TRCA':
                self.knowlayer = TRCAlayer(self.KLG_FIX, self.eeg_data_train, self.channels,self.DL)
                self.knowlayer.trca(eeg_data_train, label_data_train.reshape(-1))
                self.init1_acc = self.knowlayer.test_trca(self.eeg_data_test, self.label_data_test)
                self.output = self.knowlayer.output
                self.mixlayer = MIXlayer(mix_FIX=config['MIXlayer']['mix_FIX'], Nf=self.Nf, Nm=self.Nm,
                                         DL=self.DL)
                self.fblayer = FBlayer(FB_FIX=config[self.KLGalgorithm]['FB_FIX'], Nf=self.Nf, Nm=self.Nm, DL=self.DL)
            elif self.KLGalgorithm == 'TDCA':
                self.knowlayer = TDCAlayer(self.KLG_FIX, self.eeg_data_train, self.channels, self.DL,n_components=self.n_components)
                self.knowlayer.tdca(eeg_data_train, label_data_train.reshape(-1))
                self.init1_acc = self.knowlayer.test_tdca(self.eeg_data_test, self.label_data_test)
                self.output = self.knowlayer.output
                self.mixlayer = MIXlayer(mix_FIX=config['MIXlayer']['mix_FIX'], Nf=self.n_components, Nm=self.Nm,
                                       DL=self.DL)
                self.fblayer = FBlayer(FB_FIX=config[self.KLGalgorithm]['FB_FIX'],Nf = self.n_components,Nm= self.Nm,DL=self.DL)
            else :
                self.know1layer = TDCAlayer(True, self.eeg_data_train, self.channels, self.DL,
                                           n_components=self.n_components,output = self.output)
                self.know1layer.tdca(eeg_data_train, label_data_train.reshape(-1))
                self.init1_acc = self.know1layer.test_tdca(self.eeg_data_test, self.label_data_test)
                self.output=self.know1layer.output
                self.know2layer = TRCAlayer(self.KLG_FIX, np.transpose(
                    self.know1layer(torch.tensor(eeg_data_train, dtype=torch.float32)).detach().numpy().reshape(self.Nf, -1, self.Nm,
                                                                                 self.n_components, self.Nt),(0, 2, 3, 4, 1)),
                                            [i for i in range(self.n_components)], self.DL)

                self.know2layer.trca(
                    self.know1layer(torch.tensor(eeg_data_train, dtype=torch.float32)).detach().numpy(),
                    label_data_train.reshape(-1))

                self.init2_acc = self.know2layer.test_trca(np.transpose(
                    self.know1layer(torch.tensor(eeg_data_test,dtype =torch.float32)).detach().numpy().reshape(self.Nf, -1, self.Nm,
                                                                                 self.n_components, self.Nt),
                    (0, 2, 3, 4, 1)), self.label_data_test)


                self.knowlayer = nn.Sequential(self.know1layer,self.know2layer)
                self.mixlayer = MIXlayer(mix_FIX=config['MIXlayer']['mix_FIX'], Nf=self.Nf, Nm=self.Nm,
                                         DL=self.DL)
                self.fblayer = FBlayer(FB_FIX=config[self.KLGalgorithm]['FB_FIX'], Nf=self.Nf, Nm=self.Nm, DL=self.DL)

    def select_channel(self):
        channels = {' PZ', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', ' Oz', ' O1', ' O2'}
        with open(r'/data2/hzt/ssvep/benchmark40/64-channels.loc', mode='r') as file:
            channelsid = []
            lines = file.readlines()
            for line in lines:
                elements = [item for item in channels if item in line]
                if elements != []:
                    channelsid.append(int(line[:2]) - 1)
        return channelsid
    def select_channel_BETA(self):
        data = loadmat(os.path.join('/data2/hzt/ssvep/BETA', f'S1.mat'))
        raw_channels = data['data'][0, 0][1][0, 0][3][:, -1]
        channel_names = [ch[0].upper() for ch in raw_channels]

        def get_channel_indices(channel_names, target_channels):
            # 转为大写后匹配
            target_set = {ch.upper().strip() for ch in target_channels}
            return [i for i, ch in enumerate(channel_names) if ch in target_set]

        target_channels = {' PZ', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', ' Oz', ' O1', ' O2'}
        indices = get_channel_indices(channel_names, target_channels)
        return indices

    def forward(self,x):
        if self.Bandpass:
            x = self.Bandpasslayer(x)
        x = self.knowlayer(x)
        if self.MIX:
            x = self.mixlayer(x)
        else:
            x=x.flatten(start_dim=1, end_dim=2).unsqueeze(1)
        if self.FB:
            x = self.fblayer(x)
        else:
            x=x.flatten(start_dim=1, end_dim=2).unsqueeze(1)
        return x


class BandpassConvLayer(nn.Module):
    def __init__(self, Fs=250, initfreq = 8, deltafreq=0.2 , Nf = 40, Nm=1, kernel_size=63,Bandpass_FIX=False):
        super(BandpassConvLayer, self).__init__()
        # 计算奈奎斯特频率
        nyquist = Fs // 2
        self.Nm = Nm
        self.conv = nn.ModuleList(
            [nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2) for _ in
             range(self.Nm)])
        for j in range(self.Nm):
            nn.init.zeros_(self.conv[j].bias)
            for param in self.conv[j].parameters():
                if Bandpass_FIX:
                    param.requires_grad = False
            Wp = [initfreq*(j+1)/nyquist, (initfreq*(j+1)+deltafreq*(Nf-1))/ nyquist]  # 归一化通带频率
            Ws = [(initfreq*(j+1)-2) / nyquist,(initfreq*(j+1)+deltafreq*Nf+2)/ nyquist]  # 归一化阻带频率
            gpass = 1  # 通带最大衰减
            gstop = 40  # 阻带最小衰减
            # 设计Chebyshev滤波器
            N, Wn = signal.cheb1ord(Wp, Ws, gpass, gstop)
            N, Wn = signal.cheb1ord(Wp, Ws, gpass, gstop)
            B, A = signal.cheby1(N, gpass, Wn, btype='bandpass', analog=False)[:]

            # 计算离散系统的冲激响应
            impulse_response = signal.lfilter(B, A, np.append(1, np.zeros(kernel_size - 1)))
            # 赋值卷积核权重
            with torch.no_grad():
                self.conv[j].weight.copy_(torch.tensor(impulse_response).reshape(1, 1, -1))
    def forward(self, x):
        Nc = x.shape[1]
        if x.ndim != 3:
            raise ValueError(f'X.ndim must be 3 but got {x.shape}')
        output = torch.cat([
            torch.cat([self.conv[i](x[:,j,:].unsqueeze(1))
                for j in range(Nc)
            ], dim=1).unsqueeze(1)
            for i in range(self.Nm)
        ], dim=1)
        return output

class FBlayer(nn.Module):
    def __init__(self, FB_FIX=True, fb_coefs_a =-1.25, fb_coefs_b =0.25,Nf = 40,Nm = 1,DL=True):
        super(FBlayer, self).__init__()
        #print([math.pow(i, fb_coefs_a) + fb_coefs_b  for i in range(1, Nm + 1) for j in range(Nc)])
        self.Nm =Nm
        self.Nc= Nf
        self.DL = DL
        self.weights = nn.Parameter(
            torch.tensor([math.pow(i, fb_coefs_a) + fb_coefs_b for i in range(1, self.Nm + 1) for j in range(self.Nc)])).reshape(-1, 1)
        if FB_FIX:
            self.weights = nn.Parameter(self.weights, requires_grad=False)
        else:
            self.weights = nn.Parameter(self.weights, requires_grad=True)


    def forward(self, x):
        if self.DL:
            if x.ndim != 4: #Nh,Nm,Nf,Nt
                raise ValueError(f'X.ndim must be 4 but got {x.shape}')
            x=x.flatten(start_dim=1, end_dim=2)
            return (x * self.weights).unsqueeze(1)  # 按通道加权
        else:
            if x.ndim != 3: #Nh,Nm,Nf
                raise ValueError(f'X.ndim must be 3 but got {x.shape}')
            output = torch.cat(
                [(x[:, i, :] * self.weights[i * self.Nc:(i + 1) * self.Nc].T).unsqueeze(1) for i in range(self.Nm)],dim=1).sum(dim=1)
            return output

class MIXlayer(nn.Module):
    def __init__(self, mix_FIX=True, fb_coefs_a =-1.25, fb_coefs_b =0.25,Nf = 40,Nm = 1,DL=True):
        super(MIXlayer, self).__init__()
        #print([math.pow(i, fb_coefs_a) + fb_coefs_b  for i in range(1, Nm + 1) for j in range(Nc)])
        self.Nm =Nm
        self.Nc= Nf
        self.DL = DL
        self.weights = nn.Parameter(
            torch.tensor([math.pow(i, fb_coefs_a) + fb_coefs_b for i in range(1, self.Nm + 1) for j in range(self.Nc)])).reshape(-1, 1)
        if mix_FIX:
            self.weights = nn.Parameter(self.weights, requires_grad=False)
        else:
            self.weights = nn.Parameter(self.weights, requires_grad=True)


    def forward(self, x):
        if self.DL:
            if x.ndim != 4: #Nh,Nm,Nf,Nt
                raise ValueError(f'X.ndim must be 4 but got {x.shape}')
            a,b,c,d = x.shape
            x=x.flatten(start_dim=1, end_dim=2)
            output = (x * self.weights).unsqueeze(1)
            output = output.reshape(a,b,c,d)
            output = output.sum(dim=1, keepdim=True)
            return   output# 按通道加权
        else:
            if x.ndim != 3: #Nh,Nm,Nf
                raise ValueError(f'X.ndim must be 3 but got {x.shape}')
            output = torch.cat(
                [(x[:, i, :] * self.weights[i * self.Nc:(i + 1) * self.Nc].T).unsqueeze(1) for i in range(self.Nm)],dim=1).sum(dim=1)
            return output

class TRCAlayer(nn.Module):
    def __init__(self,KLG_FIX = True,EEGDATA=None,channels=[i for i in range(64)],DL=True):
        super(TRCAlayer, self).__init__()
        self.EEGDATA = EEGDATA
        self.DL =DL
        #print(EEGDATA.shape)
        if EEGDATA.ndim ==5:
            [self.Nf, self.Nm, self.Nc, self.T, _] = self.EEGDATA.shape
        else:
            raise ValueError(f'X.ndim must be 5 but got {EEGDATA.shape}')
        self.channels=channels
        self.n_components=1
        self.trains = np.mean(self.EEGDATA, -1)
        self.W = torch.zeros((self.Nm, self.Nf,self.Nc))
        self.is_ensemble=False
        if KLG_FIX:
            self.W = nn.Parameter(self.W,requires_grad=False)
            self.trains = nn.Parameter(torch.tensor(self.trains,dtype= torch.float32), requires_grad=False)
        else:
            self.W = nn.Parameter(self.W,requires_grad=True)
            self.trains = nn.Parameter(torch.tensor(self.trains,dtype= torch.float32), requires_grad=False)
    def forward(self,x):
        if self.DL:
            if x.ndim == 4:
                output = torch.cat([
                    torch.cat([
                        torch.matmul(x[:, i, :, :].transpose(1, 2),
                                     self.W[i, j, :].reshape(-1, 1)).transpose(1, 2)
                        for j in range(self.Nf)
                    ], dim=1).unsqueeze(1)
                    for i in range(self.Nm)
                ], dim=1)
                return output
            else:
                raise ValueError(f'X.ndim must be 4 but got {x.shape}')
        else:
            if x.ndim != self.trains.ndim:
                raise ValueError(f'X.ndim != trains.ndim')
            r = torch.zeros(( len(x) , self.Nm, self.Nf)).to('cuda:4')
            # pred = torch.zeros((n_trials, self.classes)).to(x.device)  # To store predictions
            for trial in range(len(x)):
                for class_i in range(self.Nf):
                    for fb_i in range(self.Nm): #self.W:Nm,Nf,Nc
                        if self.trains.ndim == 5:  # Nh,Nm,Nc,Nf,Nt
                            test = self.W[fb_i] @ x[trial, fb_i, :, class_i, :]  # Nf,Nt
                            train = self.W[fb_i] @ self.trains[class_i, fb_i, :, class_i, :] #Nf,Nt
                            r_tmp = torch.corrcoef(torch.stack((test[class_i], train[class_i])))[0, 1]
                            r[trial, fb_i, class_i] = r_tmp
                        else:
                            test = self.W[fb_i] @ x[trial, fb_i, :, :]  # Nf,Nt
                            train = self.W[fb_i] @ self.trains[class_i, fb_i, :, :]  # Nf,Nt

                            r_tmp = torch.corrcoef(torch.stack((test[class_i], train[class_i])))[0, 1]
                            r[ trial,fb_i, class_i] = r_tmp
            return r



    def trca(self,X=None,labels=None):
        labels = labels
        for fb_i in range (self.Nm):
            for c in range(self.Nf):
                # 提取当前类别的所有试次数据
                if X.ndim == 4:
                    X_c = X[labels == c]
                    X_c = X_c[:, fb_i, self.channels,  :]
                else:
                    raise ValueError(f'X.ndim must be 4 but got {X_c.shape}')

                n_trials, n_channels, n_samples = X_c.shape
                # 初始化协方差矩阵
                S = np.zeros((n_channels, n_channels))  # 组内协方差
                Q = np.zeros((n_channels, n_channels))  # 组间协方差
                # 计算协方差矩阵
                for i in range(n_trials):
                    Xi = X_c[i].copy()
                    Xi -= Xi.mean(axis=1,keepdims=True)  # 中心化
                    Q += Xi @ Xi.T  # 更新组间协方差

                    # 计算组内协方差（避免重复计算）
                    for j in range(i + 1, n_trials):
                        Xj = X_c[j].copy()
                        Xj -= Xj.mean(axis=1, keepdims=True)
                        S += Xi @ Xj.T + Xj @ Xi.T
                Q += 1e-8 * np.eye(n_channels)
                try:
                    S_intra_inv = np.linalg.inv(Q)  # 计算类内协方差矩阵的逆
                except np.linalg.LinAlgError:
                    # 如果无法计算逆，使用伪逆
                    S_intra_inv = np.linalg.pinv(Q)
                # 计算广义特征值问题
                eigvals, eigvecs = np.linalg.eig(S_intra_inv @ S)
                sorted_indices = np.argsort(eigvals)[::-1]  # 从大到小排序
                W = eigvecs[:, sorted_indices[:self.n_components]].T
                with torch.no_grad():
                    self.W[fb_i,c,self.channels] = torch.tensor(W,dtype=torch.float32)
        return self


    def test_trca(self, eeg,test_label):
        num_trials = eeg.shape[-1]
        fb_coefs = [math.pow(i, -1.25) + 0.25 for i in range(1, self.Nm + 1)]
        fb_coefs = np.array(fb_coefs)
        results = np.zeros((self.Nf, num_trials))
        #W=torch.tensor(torch.load(r'D:\gra_design\SSVEP-Retrain\WWW.pt'))
        for targ_i in range(self.Nf):
            if eeg.ndim==5:
                test_tmp = eeg[targ_i, :, :, :, :]
            else:
                raise ValueError(f'X.ndim must be 5 but got {eeg.shape}')
            r = np.zeros((self.Nm, self.Nf, num_trials))
            for fb_i in range(self.Nm):
                if eeg.ndim == 5:
                    testdata = test_tmp[fb_i, :, :, :]
                for class_i in range(self.Nf):
                    if self.trains.ndim == 4:
                        traindata = self.trains[class_i, fb_i, :, :]
                    else:
                        traindata = self.trains[class_i, fb_i, :, class_i,:]
                    if not self.is_ensemble:
                        w =self.W[fb_i, class_i, :]
                    else:
                        w = self.W[fb_i, :, :]

                    for trial_i in range(num_trials):
                        testdata_w = np.matmul(w.clone().detach().numpy(),testdata[:, :, trial_i])
                        traindata_w = np.matmul(w.clone().detach().numpy(),traindata[:, :].clone().detach().numpy())
                        r_tmp = np.corrcoef(testdata_w.flatten(), traindata_w.flatten())
                        r[fb_i, class_i, trial_i] = r_tmp[0, 1]

            rho = np.einsum('j, jkl -> kl', fb_coefs, r)  # (num_targs, num_trials)
            tau = np.argmax(rho, axis=0)
            results[targ_i, :] = tau
        is_correct = (results == test_label)
        is_correct = np.array(is_correct).astype(int)
        test_acc = np.mean(is_correct)
        print(test_acc)

        return test_acc

class TDCAlayer(nn.Module):
    def __init__(self,KLG_FIX = True,EEGDATA=None,channels=[i for i in range(64)],DL=True,n_components=1, output = 0 ):
        super(TDCAlayer, self).__init__()
        self.EEGDATA = EEGDATA
        self.DL =DL
        self.output = output
        #print(EEGDATA.shape)
        if EEGDATA.ndim ==5:
            [self.Nf, self.Nm, self.Nc, self.T, _] = self.EEGDATA.shape
        else:
            [self.Nf, self.Nm, self.Nc, self.Nf, self.T, _] = self.EEGDATA.shape
        self.channels=channels
        self.n_components=n_components
        self.trains = self.EEGDATA[...,0]
        self.trains = np.mean(self.EEGDATA, -1)
        self.W = torch.zeros((self.Nm, self.n_components,self.Nc))

        self.M=np.zeros((self.Nm,self.Nc,self.T))
        self.is_ensemble=False
        if KLG_FIX:
            self.W = nn.Parameter(self.W,requires_grad=False)
            self.trains = nn.Parameter(torch.tensor(self.trains,dtype= torch.float32), requires_grad=False)
        else:
            self.W = nn.Parameter(self.W,requires_grad=True)
            self.trains = nn.Parameter(torch.tensor(self.trains,dtype= torch.float32), requires_grad=False)
    def forward(self,x):
        if self.DL:
            if x.ndim == 4:
                output = torch.cat([
                    torch.cat([
                        torch.matmul(x[:, i, :, :].transpose(1, 2),
                                     self.W[i, j, :].reshape(-1, 1)).transpose(1, 2)
                        for j in range(self.n_components)
                    ], dim=1).unsqueeze(1)
                    for i in range(self.Nm)
                ], dim=1)
                # print(output.shape)
                return output
            else:
                raise ValueError(f'X.ndim must be 4 but got {x.shape}')

        else:
            if x.ndim != self.trains.ndim:
                raise ValueError(f'X.ndim != trains.ndim')
            r = torch.zeros(( len(x) , self.Nm, self.Nf)).to('cuda:4')
            # pred = torch.zeros((n_trials, self.classes)).to(x.device)  # To store predictions
            for trial in range(len(x)):
                for class_i in range(self.Nf):
                    for fb_i in range(self.Nm): #self.W:Nm,Nf,Nc
                        if self.trains.ndim == 5:  # Nh,Nm,Nc,Nf,Nt
                            test = self.W[fb_i] @ x[trial, fb_i, :, class_i, :]  # n_com,Nt
                            train = self.W[fb_i] @ self.trains[class_i, fb_i, :, class_i, :] #Nf,Nt
                            r_tmp = torch.corrcoef(torch.stack((test.flatten(), train.flatten())))[0, 1]
                            r[trial, fb_i, class_i] = r_tmp
                        else:
                            test = self.W[fb_i] @ x[trial, fb_i, :, :]  # Nf,Nt
                            train = self.W[fb_i] @ self.trains[class_i, fb_i, :, :]  # Nf,Nt

                            r_tmp = torch.corrcoef(torch.stack((test.flatten(), train.flatten())))[0, 1]
                            r[ trial,fb_i, class_i] = r_tmp
            return r



    def tdca(self,X=None,labels=None):

        for fb_i in range (self.Nm):
            X_c=X[:,fb_i,...]
            X_c = X_c - np.mean(X_c, axis=-1, keepdims=True)
            y_labels = np.unique(labels)
            if X_c.ndim == 3:
                self.M[fb_i] = np.mean(X_c, axis=0)
            if X_c.ndim == 3:
                X_c = X_c[:, self.channels, :]
            else:
                raise ValueError(f'X.ndim must be 3 but got {X_c.shape}')

            # the number of each label
            n_labels = np.array([np.sum(labels == label) for label in y_labels])
            # average template of all trials
            # class conditional template
            Ms, Ss = zip(
                *[
                    (
                        np.mean(X_c[labels == label], axis=0),
                        np.sum(
                            np.matmul(X_c[labels == label], np.swapaxes(X_c[labels == label], -1, -2)), axis=0  # Equation (2)
                        ),
                    )
                    for label in y_labels
                ]
            )
            Ms, Ss = np.stack(Ms), np.stack(Ss)
            # within-class scatter matrix
            Sw = np.sum(
                Ss - n_labels[:, np.newaxis, np.newaxis] * np.matmul(Ms, np.swapaxes(Ms, -1, -2)),
                axis=0,
            )
            Ms = Ms - self.M[fb_i,self.channels]
            # between-class scatter matrix
            Sb = np.sum(
                n_labels[:, np.newaxis, np.newaxis] * np.matmul(Ms, np.swapaxes(Ms, -1, -2)),  # Equation (3)
                axis=0,
            )

            D, W = eigh(nearestPD(Sb), nearestPD(Sw))
            ix = np.argsort(D)[::-1]  # in descending order
            D, W = D[ix], W[:, ix]
            #print(D)
            with torch.no_grad():
                self.W[fb_i,:,self.channels] = torch.tensor(W.T[:self.n_components],dtype=torch.float32)

        return self


    def test_tdca(self, eeg,test_label):
        num_trials = eeg.shape[-1]
        fb_coefs = [math.pow(i, -1.25) + 0.25 for i in range(1, self.Nm + 1)]
        fb_coefs = np.array(fb_coefs)
        results = np.zeros((self.Nf, num_trials))
        #W=torch.tensor(torch.load(r'D:\gra_design\SSVEP-Retrain\WWW.pt'))
        for targ_i in range(self.Nf):
            if eeg.ndim==5:
                test_tmp = eeg[targ_i, :, :, :, :]
            else:
                raise ValueError(f'X.ndim must be 5 but got {eeg.shape}')
            r = np.zeros((self.Nm, self.Nf, num_trials))
            for fb_i in range(self.Nm):
                testdata = test_tmp[fb_i, :, :, :]
                for class_i in range(self.Nf):
                    traindata = self.trains[class_i, fb_i, :, :]
                    #print(torch.sum(self.W[fb_i, :, self.channels]-torch.tensor(
                    #    [0.00330264, -0.00252192, -0.00396954, 0.00343817, -0.00556779, 0.0034803,
                     #    0.00282982, -0.00223608, -0.00548423],dtype=torch.float32)))
                    w = self.W[fb_i, :,]
                    #print(w)
                    for trial_i in range(num_trials):
                        if eeg.ndim==5:
                            testdata_w = np.matmul(w.clone().detach().numpy(),testdata[:, :, trial_i]-self.M[fb_i])
                        else:
                            testdata_w = np.matmul(w.clone().detach().numpy(),testdata[:, class_i, :, trial_i]-self.M[fb_i])

                        traindata_w = np.matmul(w.clone().detach().numpy(),traindata.clone().detach().numpy()-self.M[fb_i])
                        r_tmp = np.corrcoef(testdata_w.flatten(), traindata_w.flatten())
                        r[fb_i, class_i, trial_i] = r_tmp[0, 1]

            rho = np.einsum('j, jkl -> kl', fb_coefs, r)  # (num_targs, num_trials)
            self.output[targ_i] += rho
            results[targ_i, :] = np.argmax(self.output[targ_i],axis=0)
        is_correct = (results == test_label)
        is_correct = np.array(is_correct).astype(int)
        test_acc = np.mean(is_correct)
        print(test_acc)

        return test_acc





# eeg_data_train = torch.load(r'D:\gra_design\eeg_data_train.pt') #120,1,64,250
# label_data_train = torch.load(r'D:\gra_design\label_data_train.pt')
# eeg_data_test = torch.load(r'D:\gra_design\eeg_data_test.pt')
# label_data_test = torch.load(r'D:\gra_design\label_data_test.pt')
# A=Hztlayer()
#
# print(A(torch.tensor(eeg_data_train,dtype = torch.float32)).shape)
# fs=250
#
# t = np.linspace(0, 1, fs)
# # 示例输入：32个批次，3个通道，长度为256的信号
# input_signal = torch.tensor(np.sin(2 * np.pi * 8 * t)+np.sin(2 * np.pi * 16 * t),dtype=torch.float32).unsqueeze(0).unsqueeze(0)
# #print(input_signal)
# model = BandpassConvLayer()
# output = model(input_signal)
# # 绘制原始信号
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# # 计算信号的频谱（通过FFT）
# f_signal = np.fft.fft(output.detach().numpy())
# f_signal_freq = np.fft.fftfreq(len(t), 1/250)
# f_signal_magnitude = np.abs(f_signal)[0][39]
# plt.plot(f_signal_freq[:64], f_signal_magnitude[:64])
# plt.title("Original Signal")
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
#
# # 计算信号的频谱（通过FFT）
# f_signal = np.fft.fft(input_signal)
# f_signal_freq = np.fft.fftfreq(len(t), 1/250)
# f_signal_magnitude = np.abs(f_signal)[0][0]
#
# # 绘制信号的频谱
# plt.subplot(2, 1, 2)
# plt.plot(f_signal_freq[:64], f_signal_magnitude[:64])  # 只显示正频率部分
# plt.title("Original Signal Spectrum")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Magnitude")
# plt.show()
# print(output.shape)  # 查看输出形状