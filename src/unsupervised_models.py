'''
Implementation of various unsupervised rPPG models.
SOURCE: rPPG-Toolbox https://github.com/ubicomplab/rPPG-Toolbox/
See LICENSE.txt for license.
'''
import math

import numpy as np
from scipy import linalg, signal, sparse
from skimage.util import img_as_float
from sklearn.metrics import mean_squared_error



def _detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                       (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal


def _process_video(frames):
    RGB = []
    for frame in frames:
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))
    RGB = np.asarray(RGB)
    RGB = RGB.transpose(1, 0).reshape(1, 3, -1)
    return np.asarray(RGB)

def _ica(X, Nsources, Wprev=0):
    nRows = X.shape[0]
    nCols = X.shape[1]
    if nRows > nCols:
        print(
            "Warning - The number of rows is cannot be greater than the number of columns.")
        print("Please transpose input.")

    if Nsources > min(nRows, nCols):
        Nsources = min(nRows, nCols)
        print(
            'Warning - The number of soures cannot exceed number of observation channels.')
        print('The number of sources will be reduced to the number of observation channels ', Nsources)

    Winv, Zhat = _jade(X, Nsources, Wprev)
    W = np.linalg.pinv(Winv)
    return W, Zhat


def _jade(X, m, Wprev):
    n = X.shape[0]
    T = X.shape[1]
    nem = m
    seuil = 1 / math.sqrt(T) / 100
    if m < n:
        D, U = np.linalg.eig(np.matmul(X, np.mat(X).H) / T)
        Diag = D
        k = np.argsort(Diag)
        pu = Diag[k]
        ibl = np.sqrt(pu[n - m:n] - np.mean(pu[0:n - m]))
        bl = np.true_divide(np.ones(m, 1), ibl)
        W = np.matmul(np.diag(bl), np.transpose(U[0:n, k[n - m:n]]))
        IW = np.matmul(U[0:n, k[n - m:n]], np.diag(ibl))
    else:
        IW = linalg.sqrtm(np.matmul(X, X.H) / T)
        W = np.linalg.inv(IW)

    Y = np.mat(np.matmul(W, X))
    R = np.matmul(Y, Y.H) / T
    C = np.matmul(Y, Y.T) / T
    Q = np.zeros((m * m * m * m, 1))
    index = 0

    for lx in range(m):
        Y1 = Y[lx, :]
        for kx in range(m):
            Yk1 = np.multiply(Y1, np.conj(Y[kx, :]))
            for jx in range(m):
                Yjk1 = np.multiply(Yk1, np.conj(Y[jx, :]))
                for ix in range(m):
                    Q[index] = np.matmul(Yjk1 / math.sqrt(T), Y[ix, :].T / math.sqrt(
                        T)) - R[ix, jx] * R[lx, kx] - R[ix, kx] * R[lx, jx] - C[ix, lx] * np.conj(C[jx, kx])
                    index += 1
    # Compute and Reshape the significant Eigen
    D, U = np.linalg.eig(Q.reshape(m * m, m * m))
    Diag = abs(D)
    K = np.argsort(Diag)
    la = Diag[K]
    M = np.zeros((m, nem * m), dtype=complex)
    Z = np.zeros(m)
    h = m * m - 1
    for u in range(0, nem * m, m):
        Z = U[:, K[h]].reshape((m, m))
        M[:, u:u + m] = la[h] * Z
        h = h - 1
    # Approximate the Diagonalization of the Eigen Matrices:
    B = np.array([[1, 0, 0], [0, 1, 1], [0, 0 - 1j, 0 + 1j]])
    Bt = np.mat(B).H

    encore = 1
    if Wprev == 0:
        V = np.eye(m).astype(complex)
    else:
        V = np.linalg.inv(Wprev)
    # Main Loop:
    while encore:
        encore = 0
        for p in range(m - 1):
            for q in range(p + 1, m):
                Ip = np.arange(p, nem * m, m)
                Iq = np.arange(q, nem * m, m)
                g = np.mat([M[p, Ip] - M[q, Iq], M[p, Iq], M[q, Ip]])
                temp1 = np.matmul(g, g.H)
                temp2 = np.matmul(B, temp1)
                temp = np.matmul(temp2, Bt)
                D, vcp = np.linalg.eig(np.real(temp))
                K = np.argsort(D)
                la = D[K]
                angles = vcp[:, K[2]]
                if angles[0, 0] < 0:
                    angles = -angles
                c = np.sqrt(0.5 + angles[0, 0] / 2)
                s = 0.5 * (angles[1, 0] - 1j * angles[2, 0]) / c

                if abs(s) > seuil:
                    encore = 1
                    pair = [p, q]
                    G = np.mat([[c, -np.conj(s)], [s, c]])  # Givens Rotation
                    V[:, pair] = np.matmul(V[:, pair], G)
                    M[pair, :] = np.matmul(G.H, M[pair, :])
                    temp1 = c * M[:, Ip] + s * M[:, Iq]
                    temp2 = -np.conj(s) * M[:, Ip] + c * M[:, Iq]
                    temp = np.concatenate((temp1, temp2), axis=1)
                    M[:, Ip] = temp1
                    M[:, Iq] = temp2

    # Whiten the Matrix
    # Estimation of the Mixing Matrix and Signal Separation
    A = np.matmul(IW, V)
    S = np.matmul(np.mat(V).H, Y)
    return A, S


def CHROME_DEHAAN(frames, FS):
    '''
    The Chrominance Method from: De Haan, G., & Jeanne, V. (2013).
    Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
    DOI: 10.1109/TBME.2013.2266196
    ''' 
    LPF = 0.7
    HPF = 2.5
    WinSec = 1.6
    
    def process_video(frames):
        '''Calculates the average value of each frame.'''
        RGB = []
        for frame in frames:
            sum = np.sum(np.sum(frame, axis=0), axis=0)
            RGB.append(sum/(frame.shape[0]*frame.shape[1]))
        return np.asarray(RGB)

    RGB = process_video(frames)
    FN = RGB.shape[0]
    NyquistF = 1/2*FS
    B, A = signal.butter(3, [LPF/NyquistF, HPF/NyquistF], 'bandpass')

    WinL = math.ceil(WinSec*FS)
    if (WinL % 2):
        WinL = WinL+1
    NWin = math.floor((FN-WinL//2)/(WinL//2))
    WinS = 0
    WinM = int(WinS+WinL//2)
    WinE = WinS+WinL
    totallen = (WinL//2)*(NWin+1)
    S = np.zeros(totallen)

    for i in range(NWin):
        RGBBase = np.mean(RGB[WinS:WinE, :], axis=0)
        RGBNorm = np.zeros((WinE-WinS, 3))
        for temp in range(WinS, WinE):
            RGBNorm[temp-WinS] = np.true_divide(RGB[temp], RGBBase)
        Xs = np.squeeze(3*RGBNorm[:, 0]-2*RGBNorm[:, 1])
        Ys = np.squeeze(1.5*RGBNorm[:, 0]+RGBNorm[:, 1]-1.5*RGBNorm[:, 2])
        Xf = signal.filtfilt(B, A, Xs, axis=0)
        Yf = signal.filtfilt(B, A, Ys)

        Alpha = np.std(Xf) / np.std(Yf)
        SWin = Xf-Alpha*Yf
        SWin = np.multiply(SWin, signal.windows.hann(WinL))

        temp = SWin[:int(WinL//2)]
        S[WinS:WinM] = S[WinS:WinM] + SWin[:int(WinL//2)]
        S[WinM:WinE] = SWin[int(WinL//2):]
        WinS = WinM
        WinM = WinS+WinL//2
        WinE = WinS+WinL
    BVP = S
    return BVP


def PBV(frames):
    """PBV
    Improved motion robustness of remote-ppg by using the blood volume pulse signature.
    De Haan, G. & Van Leest, A.
    Physiol. measurement 35, 1913 (2014)
    """
    precessed_data = _process_video(frames)
    sig_mean = np.mean(precessed_data, axis=2)

    signal_norm_r = precessed_data[:, 0, :] / \
        np.expand_dims(sig_mean[:, 0], axis=1)
    signal_norm_g = precessed_data[:, 1, :] / \
        np.expand_dims(sig_mean[:, 1], axis=1)
    signal_norm_b = precessed_data[:, 2, :] / \
        np.expand_dims(sig_mean[:, 2], axis=1)

    pbv_n = np.array([np.std(signal_norm_r, axis=1), np.std(
        signal_norm_g, axis=1), np.std(signal_norm_b, axis=1)])
    pbv_d = np.sqrt(np.var(signal_norm_r, axis=1) +
                    np.var(signal_norm_g, axis=1) + np.var(signal_norm_b, axis=1))
    pbv = pbv_n / pbv_d

    C = np.swapaxes(
        np.array([signal_norm_r, signal_norm_g, signal_norm_b]), 0, 1)
    Ct = np.swapaxes(np.swapaxes(np.transpose(C), 0, 2), 1, 2)
    Q = np.matmul(C, Ct)
    W = np.linalg.solve(Q, np.swapaxes(pbv, 0, 1))

    A = np.matmul(Ct, np.expand_dims(W, axis=2))
    B = np.matmul(np.swapaxes(np.expand_dims(pbv.T, axis=2),
                  1, 2), np.expand_dims(W, axis=2))
    bvp = A / B
    return bvp.squeeze(axis=2).reshape(-1)


def PBV2(frames):
    precessed_data = _process_video(frames)
    data_mean = np.mean(precessed_data, axis=2)
    R_norm = precessed_data[:, 0, :] / np.expand_dims(data_mean[:, 0], axis=1)
    G_norm = precessed_data[:, 1, :] / np.expand_dims(data_mean[:, 1], axis=1)
    B_norm = precessed_data[:, 2, :] / np.expand_dims(data_mean[:, 2], axis=1)
    RGB_array = np.array([R_norm, G_norm, B_norm])

    PBV_n = np.array([np.std(R_norm, axis=1), np.std(
        G_norm, axis=1), np.std(B_norm, axis=1)])
    PBV_d = np.sqrt(np.var(R_norm, axis=1) +
                    np.var(G_norm, axis=1) + np.var(B_norm, axis=1))
    PBV = PBV_n / PBV_d
    C = np.transpose(RGB_array, (1, 0, 2))
    Ct = np.transpose(RGB_array, (1, 2, 0))

    Q = np.matmul(C, Ct)
    W = np.linalg.solve(Q, np.swapaxes(PBV, 0, 1))

    Numerator = np.matmul(Ct, np.expand_dims(W, axis=2))
    Denominator = np.matmul(np.swapaxes(np.expand_dims(
        PBV.T, axis=2), 1, 2), np.expand_dims(W, axis=2))
    BVP = Numerator / Denominator
    BVP = BVP.squeeze(axis=2).reshape(-1)
    return BVP




def POS_WANG(frames, fs):
    """POS
    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). 
    Algorithmic principles of remote PPG. 
    IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
    """
    def process_video(frames):
        '''Calculates the average value of each frame.'''
        RGB = []
        for frame in frames:
            sum = np.sum(np.sum(frame, axis=0), axis=0)
            RGB.append(sum / (frame.shape[0] * frame.shape[1]))
        return np.asarray(RGB)
    
    WinSec = 1.6
    RGB = process_video(frames)
    N = RGB.shape[0]
    H = np.zeros((1, N))
    l = math.ceil(WinSec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            Cn = np.mat(Cn).H
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = h[0, temp] - mean_h
            H[0, m:n] = H[0, m:n] + (h[0])

    BVP = H
    BVP = _detrend(np.mat(BVP).H, 100)
    BVP = np.asarray(np.transpose(BVP))[0]
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(np.double))
    return BVP


def LGI(frames):
    '''
    LGI
    Local group invariance for heart rate estimation from face videos.
    Pilz, C. S., Zaunseder, S., Krajewski, J. & Blazek, V.
    In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 1254–1262
    (2018).
    '''
    precessed_data = _process_video(frames)
    U, _, _ = np.linalg.svd(precessed_data)
    S = U[:, :, 0]
    S = np.expand_dims(S, 2)
    SST = np.matmul(S, np.swapaxes(S, 1, 2))
    p = np.tile(np.identity(3), (S.shape[0], 1, 1))
    P = p - SST
    Y = np.matmul(P, precessed_data)
    bvp = Y[:, 1, :]
    bvp = bvp.reshape(-1)
    return bvp

def ICA_POH(frames, FS):
    '''
    Non-contact, automated cardiac pulse measurements using video imaging and blind source separation.
    Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010).
    Optics express, 18(10), 10762-10774. DOI: 10.1364/OE.18.010762
    '''
    # Cut off frequency.
    LPF = 0.7
    HPF = 2.5
    
    def process_video(frames):
        '''Calculates the average value of each frame.'''
        RGB = []
        for frame in frames:
            sum = np.sum(np.sum(frame, axis=0), axis=0)
            RGB.append(sum / (frame.shape[0] * frame.shape[1]))
        return np.asarray(RGB)

    RGB = process_video(frames)

    NyquistF = 1 / 2 * FS
    BGRNorm = np.zeros(RGB.shape)
    Lambda = 100
    for c in range(3):
        BGRDetrend = _detrend(RGB[:, c], Lambda)
        BGRNorm[:, c] = (BGRDetrend - np.mean(BGRDetrend)) / np.std(BGRDetrend)
    _, S = _ica(np.mat(BGRNorm).H, 3)

    # select BVP Source
    MaxPx = np.zeros((1, 3))
    for c in range(3):
        FF = np.fft.fft(S[c, :])
        F = np.arange(0, FF.shape[1]) / FF.shape[1] * FS * 60
        FF = FF[:, 1:]
        FF = FF[0]
        N = FF.shape[0]
        Px = np.abs(FF[:math.floor(N / 2)])
        Px = np.multiply(Px, Px)
        Fx = np.arange(0, N / 2) / (N / 2) * NyquistF
        Px = Px / np.sum(Px, axis=0)
        MaxPx[0, c] = np.max(Px)
    MaxComp = np.argmax(MaxPx)
    BVP_I = S[MaxComp, :]
    B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')
    BVP_F = signal.filtfilt(B, A, np.real(BVP_I).astype(np.double))

    BVP = BVP_F[0]
    return BVP
