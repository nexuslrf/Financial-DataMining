import pywt
def wavelet_transform(data,wavelet='db2',mode='reflect'):
    (cA,cD) = pywt.dwt(data,wavelet,mode)
    return pywt.upcoef('a',cA,wavelet,take = len(data)), \
            pywt.upcoef('d',cD,wavelet,take = len(data))