import numpy as np
import matplotlib.pyplot as plt
import math, torch, time, os
import scipy.io as io



def compute_fourier_coefficients(x, y, K):
    N = len(x)
    fft_coeff = np.fft.fft(y) / N
    
    is_symmetric = (x[0] < 0) and (np.abs(x[0] + np.pi) < 1e-6)
    
    if is_symmetric:
        fft_coeff_shifted = np.fft.fftshift(fft_coeff)
        k_values = np.fft.fftshift(np.fft.fftfreq(N)) * N  # 整数k值
        pos_freq_indices = np.where(k_values > 0)[0][:K]
        # 按k值从小到大排序
        sorted_indices = np.argsort(k_values[pos_freq_indices])
        pos_freq_indices = pos_freq_indices[sorted_indices]
    else:
        fft_coeff_shifted = fft_coeff
        k_values = np.fft.fftfreq(N) * N
        pos_freq_indices = np.arange(1, K+1)
    
    a = []
    b = []
    for idx in pos_freq_indices:
        Ck = fft_coeff_shifted[idx]
        if is_symmetric:
            k = int(k_values[idx])
            phase_compensation = (-1)**(k)
            Ck *= phase_compensation
        a_k = 2 * np.real(Ck)
        b_k = -2 * np.imag(Ck) 
        a.append(a_k)
        b.append(b_k)
    
    a0 = np.real(fft_coeff[0])
    coeffs = {"a0": a0, "a": np.array(a), "b": np.array(b)}
    y_f = reconstruct_signal_from_coefficients(x, coeffs)
    error_l2 = np.linalg.norm(y_f - y) / np.linalg.norm(y)
    return coeffs, error_l2

def reconstruct_signal_from_coefficients(x, coefficients):
    N = len(x)
    a0 = coefficients["a0"]
    a = coefficients["a"]
    b = coefficients["b"]
    K = len(a)
    
    fft_coeff_recon = np.zeros(N, dtype=complex)
    fft_coeff_recon[0] = a0  # 直流分量直接放在索引0
    
    is_symmetric = (x[0] < 0) and (np.abs(x[0] + np.pi) < 1e-6)
    
    if is_symmetric:
        k_values = np.fft.fftfreq(N) * N  # 原始频率索引 [-N/2, ..., N/2-1]
        pos_freq_indices = np.where(k_values > 0)[0][:K]
        neg_freq_indices = N - pos_freq_indices
    else:
        pos_freq_indices = np.arange(1, K+1)
        neg_freq_indices = np.arange(N-1, N-K-1, -1)
    
    for i in range(K):
        if is_symmetric:
            k = int(k_values[pos_freq_indices[i]])
            phase_compensation = (-1)**k  # 相位补偿因子
            Ck_pos = (a[i] - 1j * b[i]) / 2 * phase_compensation
        else:
            Ck_pos = (a[i] - 1j * b[i]) / 2
        Ck_neg = np.conj(Ck_pos)
        fft_coeff_recon[pos_freq_indices[i]] = Ck_pos
        fft_coeff_recon[neg_freq_indices[i]] = Ck_neg
    
    y_recon = np.fft.ifft(fft_coeff_recon * N).real
    return y_recon




####################################################################测试
# N = 128  # 采样点数
# x = np.linspace(-1*np.pi, np.pi, N, endpoint=False)
# signal =  3 + 2*np.sin(x) + 4*np.sin(2*x) + 6*np.sin(3*x) +9*np.sin(4*x)+ np.sin(5*x)+ 8*np.cos(3*x)+6*np.cos(2*x) + 10*np.cos(4*x)
# # signal = 1+np.sin(x)
# test_C, error_l2 = compute_fourier_coefficients(x, signal, 5)
# yyy = reconstruct_signal_from_coefficients(x, test_C)
# plt.plot(x, yyy, '--', label='numeric')
# plt.plot(x, signal, '-', label='exact')
# error = signal-yyy
# plt.plot(x, signal-yyy, '-', label='error')


####################################################################生成数据a0 + a1*sin(x)
# 初始化设置
# np.random.seed(42)
# N = 2048
# N_COEFFS = 10  # 明确命名常数
# x = np.linspace(-np.pi, np.pi, N, endpoint=False)

# # 预分配所有傅里叶系数
# a0 = 2.0  # 明确浮点类型
# b_coeffs = np.random.rand(N_COEFFS)
# a_coeffs = np.zeros(N_COEFFS)  # 预分配所有b系数

# reconstructed_signals = np.empty((N_COEFFS, N))

# # 主循环优化
# for idx,  (b_val) in enumerate(zip( b_coeffs)):
#     coeffs = {
#         "a0": a0,
#         "a": np.array([0.0]),  
#         "b": np.array([b_val])
#     }    
#     # 信号重建
#     reconstructed_signals[idx] = reconstruct_signal_from_coefficients(x, coeffs)

# folder = './dataset_produce'    
# os.makedirs(folder, exist_ok=True)
# np.savetxt(f"{folder}/n_test.csv", reconstructed_signals)
# # np.savetxt(f"{folder}/f_coeffs.csv", b_coeffs)
    
    
    

    

















