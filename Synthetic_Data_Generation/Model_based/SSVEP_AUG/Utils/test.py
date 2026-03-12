import numpy as np
from types import SimpleNamespace

from Models.KNoW.FBCCA import FBCCA
from Models.KNoW.TRCA import TRCA
from Models.DeepL import EEGNet, CCNN, SSVEPNet, FBtCNN, ConvCA, SSVEPformer, DDGCNN
from Utils import Constraint, LossFunction, Script
from Utils.dataprocess import data_aug, MODEL_BASED_AUGS
import torch
import torch.nn as nn
import time
def test(testmethod,config,devices, train_data, train_aug, train_label, aug_label,
                   test_data,test_label):

    if testmethod == "FBCCA":
        ACC = fbcca_evaluate(config,train_data, train_aug, train_label, aug_label,
                   test_data,  test_label)
    elif testmethod == 'TRCA':
        ACC = trca_evaluate(config,train_data, train_aug, train_label, aug_label,
                   test_data,  test_label)
    else:
        ACC = model_evaluate(config,testmethod,devices,train_data,train_label,
                   test_data, test_label, train_aug, aug_label)
    return ACC
def prepare_inputs(arr):
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0, :, :]
    return np.array(arr,dtype=np.float64)      

def fbcca_evaluate(config,train_data, train_aug, train_label, aug_label,
                   test_data,  test_label):
    """
    FBCCA accuracy evaluation before/after denoising (supports tensor / numpy / TensorDataset).
    Input data must have shape [N,1,C,T] or [N,C,T].
    """

    train_X_raw = prepare_inputs(train_data)
    train_X_aug = prepare_inputs(train_aug)
    
    train_y = np.array(train_label).flatten()
    augtrain_y = np.array(aug_label).flatten()
    
    test_X = prepare_inputs(test_data)
    test_y = np.array(test_label).flatten()

    # ---------------------------
    # Load config parameters
    # ---------------------------
    datasetid = config["train_param"]['datasets']
    Fs = config[f"data_param{datasetid}"]['Fs']
    Nc = config[f"data_param{datasetid}"]['Nc']
    Nm = config["model_param"]['Nm']
    Nf = config[f"data_param{datasetid}"]['Nf']
    ws = config["train_param"]['ws']
    initfreq = config[f"data_param{datasetid}"]['initfreq']
    deltafreq = config[f"data_param{datasetid}"]['deltafreq']

    opt = SimpleNamespace(Fs=Fs, ws=ws, Nm=Nm, Nc=Nc, Nf=Nf, dataset='your_dataset',is_ensemble=False)
    targets = initfreq + np.arange(Nf) * deltafreq
    fbcca = FBCCA(opt)
    def dummy_filter_bank(eeg):
        segments, total_channels, T = eeg.shape
        assert total_channels == Nc * Nm, f"Channel count mismatch: {total_channels} != {Nc}*{Nm}"
        eeg = eeg.reshape(segments, Nm, Nc, T)
        return eeg
    fbcca.filter_bank = dummy_filter_bank
    
    pred_raw = fbcca.fbcca_classify(targets, test_X, num_harmonics=Nm,train_labels=train_y,
                                                   train_data=train_X_raw, template=True)

    pred_raw = np.array(pred_raw).flatten()
    acc_raw = np.mean(test_y== pred_raw)
    print(f"FBCCA accuracy on raw test data: {acc_raw:.4f} ({np.sum(test_y==pred_raw)}/{len(test_y)})")
    del fbcca
    
    fbcca = FBCCA(opt)
    fbcca.filter_bank = dummy_filter_bank
    pred_aug = fbcca.fbcca_classify(targets, test_X, num_harmonics=Nm,train_labels=augtrain_y,
                                                    train_data=train_X_aug, template=True)
    pred_aug = np.array(pred_aug).flatten()
    acc_aug = np.mean(test_y== pred_aug)
    print(f"FBCCA accuracy on AUG test data: {acc_aug:.4f} ({np.sum(test_y==pred_aug)}/{len(test_y)})")
    del fbcca

    return acc_raw,acc_aug

def trca_evaluate(config,train_data, train_aug, train_label, aug_label,
                   test_data,  test_label):
    train_X_raw = prepare_inputs(train_data)
    train_X_aug = prepare_inputs(train_aug)
    
    train_y = np.array(train_label).flatten()
    augtrain_y = np.array(aug_label).flatten()

    test_X = prepare_inputs(test_data)
    test_y = np.array(test_label).flatten()

    datasetid = config["train_param"]['datasets']
    Fs = config[f"data_param{datasetid}"]['Fs']
    Nc = config[f"data_param{datasetid}"]['Nc']
    Nm = config["model_param"]['Nm']
    Nf = config[f"data_param{datasetid}"]['Nf']
    ws = config["train_param"]['ws']
    Nt = int(ws*Fs)

    test_y = test_y.reshape(Nf, -1)

    train_X_raw=train_X_raw.reshape(-1, Nm, Nc, Nt)
    train_X_raw_new = np.transpose(train_X_raw.reshape(Nf, -1, Nm, Nc, Nt),
                                       (0, 2, 3, 4, 1))
    test_X = np.transpose(test_X.reshape(Nf, -1, Nm, Nc, Nt),
                                      (0, 2, 3, 4, 1))
    
    knowlayer = TRCA(train_X_raw_new)
    knowlayer.trca(train_X_raw, train_y)
    acc_raw =knowlayer.test_trca(test_X, test_y)
    print(f"TRCA accuracy on raw test data: {acc_raw:.4f} ")
    del knowlayer

    train_X_aug = train_X_aug.reshape(-1, Nm, Nc, Nt)
    train_X_aug_new = np.transpose(train_X_aug.reshape(Nf, -1, Nm, Nc, Nt),
                                     (0, 2, 3, 4, 1))
    knowlayer = TRCA(train_X_aug_new)
    knowlayer.trca(train_X_aug, augtrain_y)
    acc_aug = knowlayer.test_trca(test_X, test_y)
    print(f"TRCA accuracy on AUG test data: {acc_aug:.4f} ")
    del knowlayer


    return acc_raw,acc_aug

def model_evaluate(config,testmethod,devices,train_data, train_label,
                   test_data, test_label, aug_data=None, aug_label=None):
    train_X_raw = prepare_inputs(train_data)
    train_y = np.array(train_label).flatten()

    test_X = prepare_inputs(test_data)
    test_y = np.array(test_label).flatten()

    models1 = build_model(config,devices,testmethod)
    _,acc_raw=train_on_batch(config,models1,testmethod,devices,train_X_raw,train_y,test_X,test_y)
    del models1

    augtype = config['train_param']['augtype']
    models2 = build_model(config, devices,testmethod)
    if aug_data is not None and augtype in MODEL_BASED_AUGS:
        # For model-based augmentation, use pre-augmented data directly
        aug_X = prepare_inputs(aug_data)
        aug_y = np.array(aug_label).flatten()
        _, acc_aug = train_on_batch(config, models2, testmethod, devices, aug_X, aug_y, test_X, test_y)
    else:
        _, acc_aug = train_on_batch_AUG(config, models2,testmethod, devices, train_X_raw,train_y,test_X, test_y)
    del models2
    return acc_raw.item(),acc_aug.item()

def train_on_batch(config,Models,testmethod, device, train_data, train_label,
                   test_data, test_label,lr_jitter=False):
    datasetid = config["train_param"]['datasets']
    Es = config["model_param"]['Es']
    bz = config["train_param"]['bz']
    Fs = config[f"data_param{datasetid}"]["Fs"]  # Sample Frequency
    ws = config[f"train_param"]['ws']
    DLalgorithm = testmethod
    num_epochs = config[f"{DLalgorithm}"]['epochs']
    if DLalgorithm == "CCNN":
        train_data = CCNN.complex_spectrum_features(np.expand_dims(train_data, axis=1), FFT_PARAMS=[Fs, ws]).squeeze(1)
        test_data = CCNN.complex_spectrum_features(np.expand_dims(test_data, axis=1), FFT_PARAMS=[Fs, ws]).squeeze(1)
    train_data = torch.tensor(train_data)
    train_label = torch.tensor(train_label)
    test_data = torch.tensor(test_data)
    test_label = torch.tensor(test_label)  # classification labels as long
    
    EEGData_Train = torch.utils.data.TensorDataset(train_data.unsqueeze(1), train_label.unsqueeze(1))
    EEGData_Test = torch.utils.data.TensorDataset(test_data.unsqueeze(1), test_label.unsqueeze(1))

    # Create DataLoader for the Dataset

    train_iter = torch.utils.data.DataLoader(dataset=EEGData_Train, batch_size=bz, shuffle=True,
                                             drop_last=True)
    test_iter = torch.utils.data.DataLoader(dataset=EEGData_Test, batch_size=bz, shuffle=False,
                                            drop_last=True)

    for epoch in range(num_epochs):
        # ==================================training procedure==========================================================
        for es in range(Es):
            net = Models[es]['model']
            optimizer = Models[es]['optimizer']
            criterion = Models[es]['criterion']
            if DLalgorithm == "DDGCNN":
                lr_decay_rate = config[DLalgorithm]['lr_decay_rate']
                optim_patience = config[DLalgorithm]['optim_patience']
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=lr_decay_rate,
                                                                    patience=optim_patience, verbose=True, eps=1e-08)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_iter),
                                                                    eta_min=5e-6)
            net.train()
            sum_loss = 0.0
            sum_acc = 0.0

            for data in train_iter:
                X, y = data
                X = X.squeeze().unsqueeze(1).to(torch.float).to(device)
                y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                y_hat = net(X)

                loss = criterion(y_hat, y).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if lr_jitter and DLalgorithm != "DDGCNN":
                    scheduler.step()
                sum_loss += loss.item() / y.shape[0]
                sum_acc += (y == y_hat.argmax(dim=-1)).float().mean()

            train_loss = sum_loss / len(train_iter)
            train_acc = sum_acc / len(train_iter)
            if lr_jitter and DLalgorithm == "DDGCNN":
                scheduler.step(train_acc)

        if (epoch+1)%50 == 0:
            sum_acc = 0.0  # reset once per test_iter
            total_batches = 0

            for data in test_iter:
                total_batches += 1
                # ========== Get input ==========
                X, y = data
                X = X.type(torch.FloatTensor).to(device)
                y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)

                # ========== Ensemble output (average logits) ==========
                total_logits = 0
                for es in range(Es):
                    net = Models[es]['model']
                    net.eval()
                    with torch.no_grad():
                        logits = net(X)
                        total_logits += logits

                avg_logits = total_logits / Es
                y_pred = avg_logits.argmax(dim=-1)
                acc = (y == y_pred).float().mean()

                sum_acc += acc

            # ========== Validation accuracy ==========
            val_acc = sum_acc / total_batches
            print(f"epoch:{epoch + 1}, train_loss={train_loss:.3f}, train_acc={train_acc:.3f}, valid_acc={val_acc:.3f}")

    print(
        f'training finished at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} with final_valid_acc={val_acc:.3f}')
    torch.cuda.empty_cache()
    return Models[-1]['acc_1'],val_acc.cpu().data
def train_on_batch_AUG(config,Models,testmethod, device, train_data, train_label,
                   test_data, test_label,lr_jitter=False):
    datasetid = config["train_param"]['datasets']
    augtype=config['train_param']['augtype']
    Es = config["model_param"]['Es']
    bz = config["train_param"]['bz']
    Fs = config[f"data_param{datasetid}"]["Fs"]  # Sample Frequency
    ws = config[f"train_param"]['ws']
    DLalgorithm = testmethod
    num_epochs = config[f"{DLalgorithm}"]['epochs']
    if DLalgorithm == "CCNN":
        train_data = CCNN.complex_spectrum_features(np.expand_dims(train_data, axis=1), FFT_PARAMS=[Fs, ws]).squeeze(1)
        test_data = CCNN.complex_spectrum_features(np.expand_dims(test_data, axis=1), FFT_PARAMS=[Fs, ws]).squeeze(1)
    train_data = torch.tensor(train_data)
    train_label = torch.tensor(train_label)
    test_data = torch.tensor(test_data)
    test_label = torch.tensor(test_label)  # classification labels as long
    EEGData_Train = torch.utils.data.TensorDataset(train_data.unsqueeze(1), train_label.unsqueeze(1))
    EEGData_Test = torch.utils.data.TensorDataset(test_data.unsqueeze(1), test_label.unsqueeze(1))

    # Create DataLoader for the Dataset

    train_iter = torch.utils.data.DataLoader(dataset=EEGData_Train, batch_size=bz, shuffle=True,
                                             drop_last=True)
    test_iter = torch.utils.data.DataLoader(dataset=EEGData_Test, batch_size=bz, shuffle=False,
                                            drop_last=True)

    for epoch in range(num_epochs):
        # ==================================training procedure==========================================================
        for es in range(Es):
            net = Models[es]['model']
            optimizer = Models[es]['optimizer']
            criterion = Models[es]['criterion']
            if DLalgorithm == "DDGCNN":
                lr_decay_rate = config[DLalgorithm]['lr_decay_rate']
                optim_patience = config[DLalgorithm]['optim_patience']
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=lr_decay_rate,
                                                                    patience=optim_patience, verbose=True, eps=1e-08)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_iter),
                                                                    eta_min=5e-6)
            net.train()
            sum_loss = 0.0
            sum_acc = 0.0

            for data in train_iter:
                X, y = data
                X,y=data_aug(X,y,augtype)
                if isinstance(X, np.ndarray):
                    X = torch.from_numpy(X)

                if isinstance(y, np.ndarray):
                    y = torch.from_numpy(y)
                X = X.squeeze().unsqueeze(1).to(torch.float).to(device)
                y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                y_hat = net(X)

                loss = criterion(y_hat, y).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if lr_jitter and DLalgorithm != "DDGCNN":
                    scheduler.step()
                sum_loss += loss.item() / y.shape[0]
                sum_acc += (y == y_hat.argmax(dim=-1)).float().mean()

            train_loss = sum_loss / len(train_iter)
            train_acc = sum_acc / len(train_iter)
            if lr_jitter and DLalgorithm == "DDGCNN":
                scheduler.step(train_acc)

        if (epoch+1)%50 == 0:
            sum_acc = 0.0  # reset once per test_iter
            total_batches = 0

            for data in test_iter:
                total_batches += 1
                # ========== Get input ==========
                X, y = data
                X = X.type(torch.FloatTensor).to(device)
                y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)

                # ========== Ensemble output (average logits) ==========
                total_logits = 0
                for es in range(Es):
                    net = Models[es]['model']
                    net.eval()
                    with torch.no_grad():
                        logits = net(X)
                        total_logits += logits

                avg_logits = total_logits / Es
                y_pred = avg_logits.argmax(dim=-1)
                acc = (y == y_pred).float().mean()

                sum_acc += acc

            # ========== Validation accuracy ==========
            val_acc = sum_acc / total_batches
            print(f"epoch:{epoch + 1}, train_loss={train_loss:.3f}, train_acc={train_acc:.3f}, valid_acc={val_acc:.3f}")

    print(
        f'training finished at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} with final_valid_acc={val_acc:.3f}')
    torch.cuda.empty_cache()
    return Models[-1]['acc_1'],val_acc.cpu().data


def build_model(config,devices,testmethod,concatenate=False):
    
    '''
    Parameters
    ----------
    device: the device to save DL models
    Returns: the building model
    -------
    '''

    KLG = config['model_param']['KLG']
    DL = config['model_param']['DL']
    Nm = config["model_param"]['Nm']
    Es = config["model_param"]['Es']
    DLalgorithm = testmethod
    datasetid = config["train_param"]["datasets"]
    Nc = config[f"data_param{datasetid}"]["Nc"]  # number of channels
    Nf = config[f"data_param{datasetid}"]["Nf"]  # number of target frequency
    Fs = config[f"data_param{datasetid}"]["Fs"]  # Sample Frequency
    ws = config[f"train_param"]['ws']
    lr = config[DLalgorithm]['lr']
    wd = config[DLalgorithm]['wd']
    initfreq = config[f"data_param{datasetid}"]["initfreq"]
    deltafreq = config[f"data_param{datasetid}"]["deltafreq"]
    Nt = int(Fs * ws)
    if concatenate:
        Nt=Nt*2
    Models = []
    acc_1 = 0
    for es in range(Es):
        if datasetid >= 2:
            DLinput = 9 * max(Nm, 1)
        else:
            DLinput = Nc * max(Nm, 1)
        if DLalgorithm == "EEGNet":
            DLnet = EEGNet.EEGNet(DLinput, Nt, Nf)
        elif DLalgorithm == "CCNN":
            DLnet = CCNN.CNN(DLinput, 220, Nf)
        elif DLalgorithm == "FBtCNN":
            DLnet = FBtCNN.tCNN(DLinput, Nt, Nf, Fs)
        elif DLalgorithm == "ConvCA":
            DLnet = ConvCA.convca(DLinput, Nt, Nf)
        elif DLalgorithm == "SSVEPformer":
            DLnet = SSVEPformer.SSVEPformer(depth=2, attention_kernal_length=31, FFT_PARAMS=[Fs, ws],
                                            chs_num=DLinput, class_num=Nf,
                                            dropout=0.5, resolution=deltafreq, start_freq=initfreq, end_freq=64)
            DLnet.apply(Constraint.initialize_weights)

        elif DLalgorithm == "SSVEPNet":
            DLnet = SSVEPNet.ESNet(DLinput, Nt, Nf)
            DLnet = Constraint.Spectral_Normalization(DLnet)

        elif DLalgorithm == "DDGCNN":
            bz = config[DLalgorithm]["bz"]
            norm = config[DLalgorithm]["norm"]
            act = config[DLalgorithm]["act"]
            trans_class = config[DLalgorithm]["trans_class"]
            n_filters = config[DLalgorithm]["n_filters"]
            DLnet = DDGCNN.DenseDDGCNN([bz, Nt, DLinput], k_adj=3, num_out=n_filters, dropout=0.5, n_blocks=3,
                                       nclass=Nf,
                                       bias=False, norm=norm, act=act, trans_class=trans_class, device=devices)
        model = DLnet

        if config['train_param']["smooth"] and DLalgorithm == "SSVEPNet":
            if datasetid == 2:
                criterion = LossFunction.CELoss_Marginal_Smooth(Nf, stimulus_type=40)
            else:
                criterion = LossFunction.CELoss_Marginal_Smooth(Nf, stimulus_type=12)
        else:
            criterion = nn.CrossEntropyLoss(reduction="none")

        if DLalgorithm == "SSVEPformer":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)
        model = model.to(devices)
        Models.append({"model": model, "criterion": criterion, "optimizer": optimizer, "acc_1": acc_1})
    return Models