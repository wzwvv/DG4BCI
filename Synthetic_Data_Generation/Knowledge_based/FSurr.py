from braindecode.augmentation import FTSurrogate

def freqSur(X):
    transform = FTSurrogate(
        probability=1.,  # defines the probability of actually modifying the input
    )
    X_tr, _ = transform.operation(torch.as_tensor(X).float(), None, 1, False)  # 空间信息很重要时，设为false，0-1之间相位扰动
    return X_tr.numpy()