import numpy as np
import matplotlib.pyplot as plt
import os
import re
import scipy.signal as signal
from scipy.optimize import minimize, differential_evolution
import scipy.io.wavfile as wavfile
from typing import Literal
from read_path import load_rew_ir

# 韌體 repo (SC598) 的資料夾，second_path.h / feedback_path.h 由該 repo 的 tools/export_paths.py 產生
FIRMWARE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'EV-SC598-SOM', 'Audio_Talkthrough_I2S')

# 錄音電壓換算成數位單位用的滿刻度峰值電壓
# 板子的 ADC (ADAU1979) 單端滿刻度 2.25 V rms：除以此值即為板子的 dBFS，與 second_path.h 的 S 同一單位
# (假設 ADAU1787 J23 輸出直接接到載板 J12，前級增益與 8 月錄音時相同)
ADAU1979_FULL_SCALE_VPEAK = 2.25 * np.sqrt(2)
# 舊平台 ADAU1787 DAC 單端滿刻度 (0.5 V rms)，換算出的數值比板子單位高 13.1 dB；重現 W41 的結果時使用
ADAU1787_FULL_SCALE_VPEAK = 0.707

# 韌體 ANC 核心設定，模擬的預設值與韌體一致
# (mu、epsilon 為 2026-11-09 建議的修正值，anc_fxnlms.h 尚未更新)
FW_FS = 48000              # ANC 核心取樣率 (Hz)
FW_W_TAPS = 2048           # W(z) 階數
FW_MU = 0.01               # 搭配迴路內參考帶通 FW_REF_BAND
FW_MU_NO_FILTER = 0.002    # 不濾波時的步長：低頻佔滿 x'，0.01 在 100% duty 會在低頻發散
FW_LEAK = 1e-5             # 每個核心取樣的漏洩
FW_EPSILON = 1e-11         # 正規化分母底限 (板子單位，約為風扇關閉時 2048 * x' 功率)
FW_OUTPUT_LIMIT = 0.2      # 輸出限幅 ANC_SAFE_OUTPUT_LIMIT (100% duty 時 600 Hz 約需 0.2、3.6 kHz 約需 0.7)
FW_REF_BAND = (400, 5000)  # 迴路內參考帶通 (Hz)

def generate_input_signal(T: int = 1000, Fs: int = 10000):
    """
    產生複雜的噪音源 x(n)，包含主頻、倍頻和寬頻雜訊
    Args:
        T (int): 總樣本數
        Fs (int): 取樣頻率
    """
    n = np.arange(T)
    x_300 = np.sin(2 * np.pi * 300 * n / Fs)
    x_600 = 0.5 * np.sin(2 * np.pi * 600 * n / Fs) 
    x_900 = 0.2 * np.sin(2 * np.pi * 900 * n / Fs) 
    x_harmonic = x_300 + x_600 + x_900
    x_noise = np.random.normal(0, 0.3, T) 
    return x_harmonic + x_noise

def setup_acoustic_paths(sz_file_path: str = None, target_taps: int = 64):
    """
    建立初級路徑 P(z) 和次級路徑 S(z), S_hat(z)
    若提供 sz_file_path，則從文字檔載入真實測量數據作為 S(z)；否則使用模擬數據。
    """
    # 1. 建立初級路徑 P(z)
    P = np.zeros(target_taps)
    P[5] = 1.0  # 模擬直接路徑的延遲
    P[6] = 0.5  # 模擬聲學反射的衰減
    # 2. 建立次級路徑 S(z) 和估測路徑 S_hat(z)
    S = np.zeros(target_taps)
    if sz_file_path and os.path.exists(sz_file_path):
        # 從文字檔載入真實測量數據
        data = np.loadtxt(sz_file_path, comments='*', skiprows=15)
        
        # 尋找脈衝響應的絕對值峰值
        peak_index = np.argmax(np.abs(data))
        
        # 截取峰值前 5 個取樣點至目標長度
        start_idx = max(0, peak_index - 5)
        end_idx = start_idx + target_taps
        truncated_ir = data[start_idx:end_idx]
        
        # 若資料長度不足 target_taps，則在尾部補零
        if len(truncated_ir) < target_taps:
            truncated_ir = np.pad(truncated_ir, (0, target_taps - len(truncated_ir)))
            
        # 正規化至 [-1, 1] 區間
        S = truncated_ir / np.max(np.abs(truncated_ir))
    else:
        # 若未提供檔案或檔案不存在，則使用模擬數據
        S[2] = 1.0 # 模擬抗噪喇叭的直接路徑延遲 
        S[3] = 0.3 # 模擬抗噪喇叭的低通特性
    
    # 3. 假設估測路徑 S_hat(z) 與真實路徑 S(z) 一致
    S_hat = S.copy()
    return P, S, S_hat

class CausalSOSFilter:
    """
    逐點因果 IIR 濾波器 (二階節串接，Direct Form II Transposed)，與 scipy.signal.sosfilt 結果相同。
    用於 FxNLMS 迴路內：每個取樣點只用到現在與過去的輸入，和 DSP 即時執行的方式一致。
    """
    def __init__(self, sos: np.ndarray):
        self.sections = [tuple(float(c) for c in row) for row in np.atleast_2d(sos)]
        self.state = [[0.0, 0.0] for _ in self.sections]

    def step(self, v: float) -> float:
        """輸入一個取樣點，回傳濾波後的一個取樣點"""
        for (b0, b1, b2, _, a1, a2), z in zip(self.sections, self.state):
            out = b0 * v + z[0]
            z[0] = b1 * v - a1 * out + z[1]
            z[1] = b2 * v - a2 * out
            v = out
        return v

def run_fxnlms(x: np.ndarray,
               d: np.ndarray,
               S: np.ndarray,
               S_hat: np.ndarray,
               F_hat: np.ndarray = None,
               mu: float = FW_MU,
               filter_length: int = FW_W_TAPS,
               leak: float = FW_LEAK,
               ref_sos: np.ndarray = None,
               F: np.ndarray = None,
               eps: float = FW_EPSILON,
               clamp: float = FW_OUTPUT_LIMIT
               ):
    """
    執行具備聲學回授中和(Feedback Neutralization) 的 FxNLMS 離線演算法模擬

    物理模型：喇叭聲經回授路徑 F 傳回參考麥克風，Ref(n) = x(n) + sum_k F[k] y(n-1-k)；
    演算法再扣掉估計的回授 F_hat，u(n) = Ref(n) - sum_k F_hat[k] y(n-1-k)。
    錄音是喇叭靜音時錄的，x 不含回授，所以回授必須在模擬中加回去；只扣 F_hat 而不加 F 會憑空多出一個回授迴路。
    Args:
        x (array): 輸入訊號, 參考麥克風錄音 (喇叭靜音時錄的，不含喇叭回授)
        d (array): 目標訊號, 誤差麥克風
        S (array): 次級路徑脈衝響應 Secondary Path
        S_hat (array): 次級路徑估測脈衝響應 Estimated Secondary Path
        F_hat (array): 回授路徑估測脈衝響應 Estimated Feedback Path (tap 0 對應 y(n-1))。None 表示不做回授中和
        F (array): 真實回授路徑 (tap 0 對應 y(n-1))。None 表示與 F_hat 相同 (回授被完全抵消)；
                   給不同的值可模擬 F_hat 的誤差
        mu (float): 正規化步長 (預設與韌體相同，搭配迴路內參考帶通；不濾波時用 FW_MU_NO_FILTER)
        filter_length (int): W(z) 濾波器長度
        leak (float): 每個取樣的權重漏洩
        eps (float): 正規化分母底限，與訊號單位有關 (預設為板子單位)
        clamp (float): 輸出限幅，y 超過 ±clamp 時截斷後才送出與存入歷史 (與韌體相同)；None 表示不限幅
        ref_sos (array): 迴路內參考濾波器 (scipy sos 格式，例如 signal.butter(4, [400, 5000], 'bandpass', output='sos'))。
                         None 表示不濾波。濾波器以因果方式逐點作用在回授中和後的參考訊號，再送進 W 與 S_hat，
                         和 DSP 即時執行的位置相同。d 一律不濾波：誤差麥克風收到的是未濾波的真實聲音，
                         因此回傳的 e 與 d 可直接相減比較頻譜。
                         (不要用零相位濾波 x、d 訓練 W 後再拿未濾波的 x 計算：W 在帶外沒有被訓練，凍結後帶外會被放大)
    Returns:
        tuple: (e, y, w, x_net)
            - e (array): 殘留噪音
            - y (array): 抗噪喇叭發出的反相波
            - w (array): 最終的濾波器權重
            - x_net (array): 經過回授中和後的參考訊號
    與韌體仍有的差異：x' 功率用 filter_length 點的窗口總和 (韌體用 0.999 的 IIR 平滑)，運算為 float64 (韌體 float32)。
    Examples:
        >>> import generate_input_signal, setup_acoustic_paths, run_fxnlms, plot_results
        >>> # 模擬參數設定
        >>> T = 1000  # 總樣本數
        >>> Fs = 10000 # 取樣頻率
        >>> filter_length = 64 # 抗噪濾波器的長度
        >>> mu = 0.1 # 學習率
        >>> sz_file_path = 'IR_R_Mar_25.txt'

        >>> x = generate_input_signal(T, Fs)
        >>> P, S, S_hat = setup_acoustic_paths()
        >>> e, y, w, x_net = run_fxnlms(x, d, S, S_hat, F_hat, mu, filter_length)
        >>> plot_results(x, e, Fs)
    """
    # TODO(速度，暫不處理)：每個取樣都整列平移 x_history / x_prime_history / y_history (O(N) 複製)，
    #       改成環狀緩衝區約可快 5 倍；離線模擬不要求速度
    T = len(x)
    n_clipped = 0               # 輸出被限幅的取樣數
    w = np.zeros(filter_length) # 抗噪濾波器 W(z) 
    e = np.zeros(T)             # 儲存殘留噪音 (Error)
    y = np.zeros(T)             # 儲存抗噪喇叭發出的反相波
    x_net = np.zeros(T)         # 儲存經過回授中和後的參考訊號

    # 為了計算卷積，我們需要歷史記憶體 (Buffers)
    x_history = np.zeros(max(filter_length, len(S_hat)))
    x_prime_history = np.zeros(filter_length)
    if F_hat is None:
        F_hat = np.zeros(1)
    if F is None:
        F = F_hat
    # F 與 F_hat 相同時，加回的回授與扣掉的估計完全抵消，參考訊號就是錄音 x
    feedback_cancels = len(F) == len(F_hat) and np.array_equal(F, F_hat)
    y_history = np.zeros(max(len(S), len(F_hat), len(F)))
    ref_filter = CausalSOSFilter(ref_sos) if ref_sos is not None else None

    for i in range(T):
        # a. 喇叭回授 (物理) 與回授中和 (Feedback Neutralization)：y_history[0] 此時為 y(n-1)
        if feedback_cancels:
            x_net[i] = x[i]
        else:
            ref_mic = x[i] + np.dot(F, y_history[:len(F)])
            x_net[i] = ref_mic - np.dot(F_hat, y_history[:len(F_hat)])

        # a2. 迴路內參考濾波 (因果 IIR)：在回授中和之後、送進 W 與 S_hat 之前，與 DSP 相同
        u = ref_filter.step(x_net[i]) if ref_filter is not None else x_net[i]

        # 更新歷史參考訊號
        x_history[1:] = x_history[:-1]
        x_history[0] = u
        
        # b. 計算 Filtered-x (也就是 x_prime)：將參考訊號 x 通過 S_hat
        x_prime = np.dot(S_hat, x_history[:len(S_hat)])
        
        # 更新 Filtered-x 歷史紀錄
        x_prime_history[1:] = x_prime_history[:-1]
        x_prime_history[0] = x_prime
        
        # c. 計算抗噪喇叭的輸出 y(n)
        y[i] = np.dot(w, x_history[:filter_length]) # 抗噪濾波器 W(z) 的輸出
        # 輸出限幅：與韌體相同，先截斷再送到喇叭並存入回授歷史
        if clamp is not None and abs(y[i]) > clamp:
            y[i] = np.sign(y[i]) * clamp
            n_clipped += 1
        
        # 更新喇叭輸出歷史紀錄
        y_history[1:] = y_history[:-1]
        y_history[0] = y[i]
        
        # d. 真實世界的物理疊加：麥克風收到的殘差 e(n) = 噪音經過 P + 抗噪波經過 S
        antinoise_at_mic = np.dot(S, y_history[:len(S)]) # 抗噪波經過次級路徑 S(z) 到達誤差麥克風
        e[i] = d[i] + antinoise_at_mic # 注意：演算法會讓 y 自動變成反相，所以這裡是相加
        
        # e. NLMS 權重更新 (Normalized Weight Update)
        # eps: 正規化分母底限 (參數，與訊號單位有關)
        
        # 計算 Filtered-x 歷史陣列的當下總能量 (也就是所有元素的平方和)
        power = np.dot(x_prime_history, x_prime_history) 
        
        # 使用正規化公式更新權重
        # w = w - mu * e[i] * x_prime_history
        w = w * (1 - leak) - (mu / (power + eps)) * e[i] * x_prime_history

    # 計算收斂後的降噪量
    d_power = np.mean(d[-10000:]**2)  # 取最後 10000 個樣本的平均功率
    e_power = np.mean(e[-10000:]**2)  # 取最後 10000 個樣本的平均功率
    attenuation_db = 10 * np.log10(d_power / e_power) if e_power > 0 else float('inf')
    print(f"最終收斂後的降噪量: {attenuation_db:.2f} dB, 輸出限幅比例: {100 * n_clipped / max(T, 1):.1f}%")

    return (e, y, w, x_net)

def plot_time_domain_residual_comparison(d: np.ndarray, e: np.ndarray, fs: int = 48000, last_n: int = None):
    '''
    時域波形比較圖，顯示原始噪音 d(n) 與 FxNLMS 後的殘留噪音 e(n)

    Args:
        d (array): 原始噪音訊號
        e (array): FxNLMS 後的殘留噪音
        fs (int): 取樣頻率
        last_n (int): 顯示最後 N 個樣本的結果
    '''
    if len(d) != len(e):
        raise ValueError("d 與 e 的長度必須相同")
    t = np.arange(len(d)) / fs
    if last_n is not None:
        t = t[-last_n:]
        d = d[-last_n:]
        e = e[-last_n:]
    plt.figure(figsize=(10, 6))
    plt.plot(t, d, label='Original Noise d(n)', alpha=0.5)
    plt.plot(t, e, label='Residual Noise e(n) after ANC', color='red', alpha=0.7)
    plt.title('Time-domain Residual Comparison (Last {} samples)'.format(last_n if last_n else len(d)))
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def convert_fir_to_biquad(w_fir: np.ndarray, eval_point:int = 2048, original_fs: int = 48000, target_fs: int = 48000, num_biquads: int = 4):
    """
    將 FIR 濾波器權重轉換為多個串聯的 Biquad (SOS) 係數，並從原始fs轉換為目標fs。
    解決了 FIR 權重過長導致的 IIR 擬合不良問題，並且可以針對特定頻段進行優化。
    不需要在時域上進行插值，直接在頻域上進行擬合，避免了時域插值可能引入的相位失真。
    
    Args:
        w_fir (array): 由 FxNLMS 算出的 FIR 權重陣列
        eval_point (int): 評估點數
        original_fs (int): 原始取樣頻率
        target_fs (int): 目標取樣頻率
        num_biquads (int): 預期擬合的 Biquad 數量。1 個 Biquad = 2 階 IIR。
                            (例如 4 個 Biquad 代表使用 8 階 IIR 濾波器)
    
    Returns:
        sos (ndarray): 形狀為 (num_biquads, 6) 的陣列，
                   每一列包含 [b0, b1, b2, a0, a1, a2]
    
    Examples:
        >>> # 建立一組虛擬的 FIR 權重 (實務上這裡代入您的 FxNLMS 收斂結果 w)
        >>> np.random.seed(42)
        >>> w_mock = np.random.normal(0, 0.1, 64) 
        >>> w_mock[5] = 1.0  # 模擬一個主要脈衝
        >>> # 轉換為 4 個 Biquad (8 階 IIR)
        >>> sos_matrix = convert_fir_to_biquad(w_mock, num_biquads=4)
    """
    print(f"{original_fs} Hz FIR 權重轉換為 {num_biquads} 個 Biquad (SOS) 係數，目標取樣率: {target_fs} Hz")

    # 找出 FIR 能量最大的點，這通常代表系統的主要物理延遲
    peak_idx = np.argmax(np.abs(w_fir))
    print(f"偵測到 FIR 主要物理延遲為 {peak_idx} samples (在 {original_fs}Hz 下)")

    # 1. 計算 FIR 濾波器的複數頻率響應
    # 在 0 到 Nyquist 頻率之間取 eval_point 個點
    # 注意：傳入 fs 參數後，回傳的 freq_hz 已經是 Hz 單位
    freqs_hz, h_target_raw = signal.freqz(w_fir, worN=eval_point, fs=original_fs)
    
    # 在頻域中「倒轉」這個時間延遲，強制將目標頻率響應的相位拉平 (Minimum-phase like)
    # 數學原理：相移 = e^(j * w * delay)
    # 將 Hz 轉換回正規化角頻率 (radians/sample) 才能正確計算延遲相移
    w_rad = freqs_hz * 2 * np.pi / original_fs
    phase_compensation = np.exp(1j * w_rad * peak_idx)
    h_target_aligned = h_target_raw * phase_compensation
    
    # 2. 建立目標頻段 (例如 100 到 12 kHz) 
    valid_idx = (freqs_hz >= 100) & (freqs_hz <= 12000)
    target_freqs = freqs_hz[valid_idx]
    h_target = h_target_aligned[valid_idx]

    # 3. 建立權重遮罩，只針對基頻和倍頻
    weight_mask = np.ones_like(target_freqs) * 0.01  # 預設低權重
    
    harmonic_freqs = [600, 3600]  # 假設主要噪音源的基頻與倍頻
    bandwidth = 150  # 每個頻段的寬度

    for hf in harmonic_freqs:
        # 設定權重遮罩，提高基頻和倍頻的影響
        hf_idx = (target_freqs >= hf - bandwidth) & (target_freqs <= hf + bandwidth)
        weight_mask[hf_idx] = 10.0

    # 4. 使用 fit_iir_minimize 進行 IIR 擬合
    # 回傳sos矩陣，形狀為 (num_biquads, 6)，每列包含 [b0, b1, b2, a0, a1, a2]
    sos_opt = fit_iir_minimize(target_freqs, h_target, num_biquads, target_fs, weight_mask)
    
    # 5. 視覺化驗證(加上權重遮罩的標示)
    _, h_opt_target_fs = signal.freqz_sos(sos_opt, worN=target_freqs, fs=target_fs)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(target_freqs, 20 * np.log10(np.abs(h_target) + 1e-12), label=f'Target FIR ({original_fs}Hz, Delay Removed)')
    ax1.plot(target_freqs, 20 * np.log10(np.abs(h_opt_target_fs) + 1e-12), '--', label=f'Fitted Biquad ({target_fs}Hz)')
    ax1.fill_between(target_freqs, -60, 10, where=(weight_mask > 1), color='orange', alpha=0.2, label='High Weight Zone')
    ax1.set_title(f'Magnitude Response Mapping (Harmonic Focused), Requires extra {peak_idx} samples delay block in DSP')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.legend()
    ax1.grid()
    
    ax2.plot(target_freqs, np.unwrap(np.angle(h_target)), label=f'Target Phase ({original_fs}Hz), Delay Removed')
    ax2.plot(target_freqs, np.unwrap(np.angle(h_opt_target_fs)), '--', label=f'Fitted Phase ({target_fs}Hz)')
    ax2.fill_between(target_freqs, -10, 10, where=(weight_mask > 1), color='orange', alpha=0.2)
    ax2.set_title('Phase Response Mapping')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (rad)')
    ax2.legend()
    ax2.grid()
    
    plt.tight_layout()
    plt.show()

    return sos_opt

def print_sigmastudio_coefficients(sos):
    """
    將 SOS 矩陣印出為符合 SigmaStudio Biquad 輸入格式的文字。
    注意：SigmaStudio 的 IIR 方程式分母通常定義為 1 - a1*z^-1 - a2*z^-2，
    而 Scipy 定義為 1 + a1*z^-1 + a2*z^-2，因此 a1 與 a2 需變號 (乘上 -1)。
    """
    print("=== Biquad Coefficients for SigmaStudio ===")
    for i, section in enumerate(sos):
        b0, b1, b2, a0, a1, a2 = section
        
        # 正規化確保 a0 = 1 (tf2sos 預設已處理，此為二次確認)
        b0, b1, b2 = b0/a0, b1/a0, b2/a0
        a1, a2 = a1/a0, a2/a0
        
        # ADI DSP 分母係數變號處理
        a1_dsp = -a1
        a2_dsp = -a2
        
        print(f"Biquad {i+1}:")
        print(f"  b0 = {b0:.8f}")
        print(f"  b1 = {b1:.8f}")
        print(f"  b2 = {b2:.8f}")
        print(f"  a1 = {a1_dsp:.8f} (Inverted for ADI)")
        print(f"  a2 = {a2_dsp:.8f} (Inverted for ADI)")
        print("-" * 40)

def loss_function(params: np.ndarray, w_norm: np.ndarray, h_target: np.ndarray, num_biquads: int, target_fs: int = 48000, weight_mask: np.ndarray = None):
    """
    定義一個損失函數，用於優化 Biquad 係數以擬合目標頻率響應。
    
    Args:
        params (array): Biquad 係數陣列，去除a0
        w_norm (array): 正規化頻率點陣列，對應於 h_target 的頻率響應
        h_target (array): 目標頻率響應，來自原始 FIR 權重的 freqz 結果
        num_biquads (int): Biquad 區段數量
        target_fs (int): 目標取樣頻率
        weight_mask (array): 加權遮罩，用於調整不同頻率點的影響權重

    Returns:
        loss (float): 頻率響應擬合的損失值 (例如 MSE)
    """
    params_2d = params.reshape((num_biquads, 5)) # 每個 Biquad 有 5 個參數 (b0, b1, b2, a1, a2)
    sos = np.insert(params_2d, 3, 1.0, axis=1) # 在每列的第4個位置插入 a0=1.0
    
    # 計算 IIR 激波器的頻率響應
    # freq_sos 如果產生 RuntimeWarning 給予極大懲罰值
    try: 
        w, h_iir = signal.freqz_sos(sos, worN=w_norm, fs=target_fs)
    except Exception:
        return 1e6
    
    # 避免 RuntimeWarning 中 NaN 或 Inf 導致的損失計算錯誤
    if np.any(np.isnan(h_iir)) or np.any(np.isinf(h_iir)):
        return 1e6  # 給予極大懲罰值，避免優化器陷入錯誤解
    
    # 計算損失 (例如 Mean Squared Error) 並乘上權重遮罩
    if weight_mask is None or len(weight_mask) != len(h_target):
        weight_mask = np.ones_like(h_target)  # 若未提供遮罩，則使用全 1 的遮罩

    #loss = np.mean(np.abs(h_iir - h_target)**2 * weight_mask) # 計算複數誤差(考慮振幅與相位)
    #return loss
    # 也可以考慮其他損失函數，例如頻率響應的相位差異或加權損失等，根據實際需求調整。
    
    # 1. 振幅誤差 (dB 級別比較，避免線性尺度的極端值主導)
    mag_target_db = 20 * np.log10(np.abs(h_target) + 1e-12)
    mag_iir_db = 20 * np.log10(np.abs(h_iir) + 1e-12)
    mag_loss = np.mean(weight_mask * (mag_target_db - mag_iir_db)**2)
    
    # 2. 相位誤差 (弧度比較，使用 unwrap 避免 2pi 纏繞問題)
    phase_target = np.unwrap(np.angle(h_target))
    phase_iir = np.unwrap(np.angle(h_iir))
    phase_loss = np.mean(weight_mask * (phase_target - phase_iir)**2)
    
    # 總和損失 (可微調振幅與相位的佔比)
    return float(mag_loss + 10.0 * phase_loss)

def fit_iir_minimize(w_norm_target, h_target, num_biquads, target_fs=48000, weight_mask=None):
    """
    使用 DE (全域搜尋) + L-BFGS-B (局部微調)
    尋找最佳的 Biquad 係數，以擬合目標頻率響應。
    
    Args:
        w_norm_target (array): 正規化頻率點陣列，對應於 h_target 的頻率響應
        h_target (array): 目標頻率響應，來自原始 FIR 權重的 freqz 結果
        num_biquads (int): Biquad 區段數量
    
    Returns:
        sos_opt (ndarray): 最佳化後的 SOS 矩陣
    
    """
    # 1. 建立邊界條件
    # 每個 Biquad 5 個變數: (b0, b1, b2, a1, a2)
    # 設定 Jury 穩定性準則邊界 (加上安全裕度，避免極點過於貼近單位圓)
    bounds = []
    for _ in range(num_biquads):
        bounds.extend([
            (-2.0, 2.0),    # b0
            (-2.0, 2.0),    # b1
            (-2.0, 2.0),    # b2
            (-1.99, 1.99),  # a1 (理論極限為 -2 到 2)
            (-0.99, 0.99)   # a2 (理論極限為 -1 到 1)
        ])
    # 2. 階段一：差分進化演算法(Differential Evolution)全域搜尋
    # popsize (族群大小) 與 maxiter (最大迭代次數) 可以根據問題複雜度調整
    de_result = differential_evolution(
        loss_function, 
        bounds=bounds, 
        args=(w_norm_target, h_target, num_biquads, target_fs, weight_mask), 
        strategy='best1bin', 
        maxiter=1000,
        popsize=20,
        tol=1e-3,
        disp=False  # 顯示優化過程的訊息 (可設為 True 以觀察優化進度)
    )

    # 3. 階段二：局部優化 (L-BFGS-B) 精細調整
    # 使用差分進化的結果作為初始值，進行局部優化以進一步降低損失
    result = minimize(
        loss_function, 
        x0=de_result.x, 
        args=(w_norm_target, h_target, num_biquads, target_fs, weight_mask), 
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500}  # 設定最大迭代次數與顯示選項
    )
    
    # 4. 從優化結果中重建 SOS 矩陣
    opt_params = result.x if result.success else de_result.x
    opt_params_2d = opt_params.reshape((num_biquads, 5)) # 每個 Biquad 有 5 個參數 (b0, b1, b2, a1, a2)
    sos_opt = np.insert(opt_params_2d, 3, 1.0, axis=1) # 在每列的第4個位置插入 a0=1.0
    print(f"Optimization Success: {result.success}, Final Loss: {result.fun:.6f}")
    return sos_opt

def load_rew_ir_and_denoise(file_path: str, target_taps: int = 512, pre_peak_margin: int = 10, taper_ratio: float = 0.1, visualize: bool = False) -> np.ndarray:
    """
    載入 REW 匯出的 TXT 脈衝響應，底層調用 read_path.load_rew_ir 進行截斷與尾部去底噪 (Windowing)。

    Args:
        file_path (str): REW 匯出的 TXT 檔案路徑
        target_taps (int): 目標 tap 數量
        pre_peak_margin (int): 峰值前的邊緣距離
        taper_ratio (float): 尾部平滑衰減的樣本數比例
        visualize (bool): 是否顯示視覺化結果

    Returns:
        denoised_ir (np.ndarray): 去底噪後的脈衝響應陣列
    """
    denoised_list, meta = load_rew_ir(
        filepath=file_path,
        target_taps=target_taps,
        pre_peak_margin=pre_peak_margin,
        taper_ratio=taper_ratio
    )
    denoised_ir = np.array(denoised_list, dtype=np.float64)

    if visualize:
        truncated_ir = np.array(meta.get("truncated_ir", denoised_list), dtype=np.float64)
        plt.figure(figsize=(10, 4))
        plt.plot(truncated_ir, label="Truncated IR (Before Denoising)", alpha=0.5)
        plt.plot(denoised_ir, label="Denoised IR (After Windowing)", color='red')
        filename = os.path.basename(file_path)
        plt.title(f"{filename} Pre-processing (Peak: {meta.get('peak_index', 0)}, Margin: {pre_peak_margin}, Taper: {taper_ratio*100:.0f}%)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return denoised_ir

def load_anc_path_header(file_path: str) -> tuple:
    """
    讀取韌體的路徑標頭檔 (second_path.h / feedback_path.h)。
    這兩個檔由 tools/export_paths.py 從板子經 48 kHz 核心邊界的量測產生，已包含 ADC/DAC、取樣率濾波器與
    DMA 的全部延遲，直接使用即可，不要再從峰值前截取 (REW 的 IR 截取會把真實延遲拿掉)。

    tap 定義與 run_fxnlms 相同：
        - SECOND_PATH[k]：核心輸出 y(n-k) 對核心 Err(n) 的響應 (k = 0 為同一個取樣點)
        - FEEDBACK_PATH[k]：y(n-1-k) 對核心 Ref(n) 的響應 (韌體在算出 y(n) 之前先扣回授，所以從 y(n-1) 開始)

    Args:
        file_path (str): 標頭檔路徑，例如 os.path.join(FIRMWARE_DIR, 'second_path.h')

    Returns:
        tuple: (taps, fs)
            - taps (np.ndarray): 路徑脈衝響應係數
            - fs (int): 取樣率 (ANC 核心取樣率)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    array = re.search(r'static\s+const\s+float\s+\w+\s*\[[^\]]*\]\s*=\s*\{(.*?)\}\s*;', text, re.S)
    n_taps = re.search(r'#define\s+\w+_TAPS\s+\(?(\d+)\)?', text)
    fs = re.search(r'#define\s+\w+_TAPS_FS\s+\(\(uint32_t\)\s*(\d+)\)', text)
    if array is None or n_taps is None or fs is None:
        raise ValueError(f"無法解析路徑標頭檔: {file_path}")
    taps = np.array([float(v.rstrip('fF')) for v in
                     re.findall(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?[fF]?', array.group(1))])
    if len(taps) != int(n_taps.group(1)):
        raise ValueError(f"{file_path}: 係數 {len(taps)} 個，與定義的 {n_taps.group(1)} 不符")
    return taps, int(fs.group(1))

def load_and_scale_dsp_dac_output_data(file_path: str, dsp_full_scale_voltage: float = ADAU1979_FULL_SCALE_VPEAK):
    """
    讀取量測數據，並依據 DSP 硬體滿載電壓將其縮放至 [-1, 1]

    Args:
        file_path (str): 檔案路徑 (.npz 或 .wav)
        dsp_full_scale_voltage (float): 換算成數位單位用的滿載峰值電壓。
                                        錄音是 ADAU1787 J23 輸出的電壓，現在由載板 J12 的 ADAU1979 數位化，
                                        單端滿載 2.25 V rms = 3.18 V peak (預設 ADAU1979_FULL_SCALE_VPEAK)，
                                        換算後與 second_path.h 的 S、韌體的限幅與 epsilon 同一單位。
                                        舊平台 ADAU1787 DAC 差動滿載 1 V rms，J23 單端為 0.5 V rms = 0.707 V peak
                                        (ADAU1787_FULL_SCALE_VPEAK)，數值比板子單位高 13.1 dB；重現 W41 的結果時使用。
    
    Returns:
        tuple: (x_scaled, d_scaled, fs, metadata)
            - x_scaled (np.ndarray): 縮放後的參考訊號陣列。
            - d_scaled (np.ndarray): 縮放後的誤差訊號陣列。
            - fs (float): 取樣頻率。
            - metadata (dict): 包含原始資料型態與縮放比例的詮釋資料。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到檔案: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    metadata = {'dsp_full_scale_voltage': dsp_full_scale_voltage, 'original_dtype': None}
    
    if ext == '.npz':
        data = np.load(file_path)
        if 'x' not in data or 'd' not in data or 'fs' not in data:
            raise ValueError("NPZ 檔案必須包含 'x', 'd', 'fs' 三個欄位")
        x_raw = data['x']
        d_raw = data['d']
        fs = float(data['fs'])

        metadata['original_dtype'] = str(x_raw.dtype)
        metadata['source_format'] = 'npz_voltage'

        # 將真實電壓除以 DSP 滿載電壓，映射至數位 0 dBFS
        x_scaled = x_raw / dsp_full_scale_voltage
        d_scaled = d_raw / dsp_full_scale_voltage
    
    elif ext == '.wav':    
        fs_int, data = wavfile.read(file_path)
        fs = float(fs_int)
        metadata['original_dtype'] = str(data.dtype)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("WAV 檔案必須為雙聲道 (Stereo) 格式")
        
        x_raw = data[:, 0]  # 取第一個聲道作為參考訊號
        d_raw = data[:, 1]  # 取第二個聲道作為誤差訊號

        # 判斷 WAV 編碼型態並進行轉換
        if np.issubdtype(data.dtype, np.integer):
            metadata['source_format'] = 'wav_integer'
            print("資料型態為整數，將進行正規化處理...")
            max_val = float(np.abs(np.iinfo(data.dtype).min))
            x_scaled = x_raw.astype(np.float64) / max_val
            d_scaled = d_raw.astype(np.float64) / max_val
        elif np.issubdtype(data.dtype, np.floating):
            metadata['source_format'] = 'wav_float'
            # 儲存時為浮點數電壓，執行與 NPZ 相同的硬體電壓映射
            x_scaled = x_raw / dsp_full_scale_voltage
            d_scaled = d_raw / dsp_full_scale_voltage
        else:
            raise ValueError(f"未知資料型態: {data.dtype}")
    
    else:
        raise ValueError(f"不支援的檔案格式: {ext}. 只支援 .npz 或 .wav")
    
    # 檢查是否產生嚴重削波 (即量測電壓大於 DSP 滿載電壓)
    max_amp = max(np.max(np.abs(x_scaled)), np.max(np.abs(d_scaled)))
    if max_amp > 1.0:
        print(f"警告：輸入訊號峰值比例為 {max_amp:.3f}。")
        print("實體量測電壓已超過設定的 DSP DAC 單端滿載電壓，硬體訊號已發生削波 (Clipping) 失真。")
    
    return x_scaled, d_scaled, fs, metadata

def create_simulated_primary_path(S: np.ndarray, delay_samples: int = 20, cutoff_freq: float = 2000, Fs: int = 48000):
    """
    建立模擬的初級路徑 P(z)，假設為一個簡單的延遲與衰減。
    
    Args:
        S (np.ndarray): 次級路徑的脈衝響應，用於決定初級路徑的延遲
        delay_samples (int): 延遲樣本數，預設為 20
        attenuation (float): 衰減係數，預設為 0.8
        Fs (int): 取樣率，預設為 48000
        cutoff_freq (float): 低通濾波器的截止頻率，預設為 2000 Hz
    
    Returns:
        P (np.ndarray): 模擬的初級路徑脈衝響應
    """
    # 1. Time shifting and cutting: 將 S(z) 延遲 delay_samples，並截斷至與 S(z) 相同長度
    P_delayed = np.pad(S, (delay_samples, 0), mode='constant', constant_values=0)[:len(S)]
    
    # 2. Apply a low-pass filter to simulate the frequency response of the primary path
    nyquist = 0.5 * Fs
    normal_cutoff = cutoff_freq / nyquist
    sos = signal.butter(N=4, Wn=normal_cutoff, btype='low', analog=False, output='sos')
    P = signal.sosfilt(sos, P_delayed)
    return P

def analyze_coherence(x: np.ndarray, d: np.ndarray, fs: int = 48000, nfft: int = 4096):
    """
    計算參考麥克風訊號 x(n) 與誤差麥克風訊號 d(n) 的相干函數 (Coherence Function)
    
    Args:
        x (np.ndarray): 參考麥克風訊號
        d (np.ndarray): 誤差麥克風訊號
        fs (int): 取樣率
        nfft (int): FFT 點數
    
    Returns:
        tuple: (f, Cxy, A_max)
            - f (np.ndarray): 頻率軸
            - Cxy (np.ndarray): 相干函數值    
            - A_max (np.ndarray): 理論最大衰減量
    """
    min_len = min(len(x), len(d))
    x = x[:min_len]
    d = d[:min_len]
    f, Cxy = signal.coherence(x, d, fs=fs, window='hann', nperseg=nfft, noverlap=nfft//2)
    Cxy = np.clip(Cxy, 1e-12, 0.999999999) # 避免 log(0) 的情況

    # 計算理論最大衰減量 (A_max) = -10 * log10(1 - gamma^2)，其中 gamma 為相干函數值
    A_max = -10 * np.log10(1 - Cxy)

    return f, Cxy, A_max

def plot_coherence_analysis(f: np.ndarray, Cxy: np.ndarray, A_max: np.ndarray, label: str = 'Data'):
    """
    繪製相干函數與理論最大降噪深度之雙子圖。
    供單次呼叫或多次疊加比較使用。
    """
    if not plt.get_fignums():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.canvas.manager.set_window_title('ANC Coherence Analysis')
    else:
        fig = plt.gcf()
        ax1, ax2 = fig.axes
    # 1. 繪製相干性 (線性 Y 軸 0.0 ~ 1.0)
    ax1.semilogx(f, Cxy, label=label, linewidth=1.5)
    ax1.set_ylabel('Coherence $\\gamma^2$ (Linear)')
    ax1.set_title('Magnitude-Squared Coherence')
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, which="major", ls="-", alpha=0.8)
    ax1.grid(True, which="minor", ls="--", alpha=0.4)
    ax1.legend(loc='lower right')
    
    # 2. 繪製最大降噪深度 (線性 Y 軸 0 ~ 40 dB)
    ax2.semilogx(f, A_max, label=label, linewidth=1.5)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Max Attenuation (dB)')
    ax2.set_title('Theoretical Maximum ANC Attenuation')
    ax2.set_ylim([0, 40])
    ax2.set_xlim([20, 20000])
    ax2.grid(True, which="major", ls="-", alpha=0.8)
    ax2.grid(True, which="minor", ls="--", alpha=0.4)
    
    plt.tight_layout()
    plt.show()

def resample_data(data: np.ndarray, original_fs: float, target_fs: float, visualize: bool = False) -> np.ndarray:
    """
    使用 scipy.signal.resample_poly 多相濾波器(polyphase filter)
    傳統的 scipy.signal.resample (基於 FFT) 會在資料前後端產生環振 (Ringing) 與相位失真。多相濾波器能維持較佳的線性相位特性
    將數據重新取樣至目標取樣頻率。
    
    Args:
        data (np.ndarray): 原始數據陣列。
        original_fs (float): 原始取樣頻率。
        target_fs (float): 目標取樣頻率。
        visualize (bool): 是否顯示重新取樣前後的波形比較。

    Returns:
        np.ndarray: 重新取樣後的數據陣列。
    """
    if original_fs == target_fs:
        return data  # 若取樣率相同，直接返回原始數據

    # 計算重取樣的上下取樣率
    gcd = np.gcd(int(original_fs), int(target_fs))
    up = int(target_fs // gcd)
    down = int(original_fs // gcd)

    resampled_data = signal.resample_poly(data, up, down)
    print(f"Resampled: {len(resampled_data)} samples", f" from {len(data)} samples at {original_fs} Hz to {target_fs} Hz")

    if visualize:
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(data[:1000])  # 只顯示前 1000 個樣本以便觀察
        plt.title('Original Data, Sample Rate: {} Hz'.format(original_fs))
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(resampled_data[:1000*up//down])  # 只顯示前 1000 個樣本以便觀察
        plt.title('Resampled Data, Sample Rate: {} Hz'.format(target_fs))
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return resampled_data

def run_anc_simulation(x: np.ndarray, rew_file_path: str, target_taps: int = 512, num_biquads: int = 4, mu: float = 0.01, filter_length: int = 256):
    """
    執行 ANC 模擬流程，包含載入 WAV 檔案、次級路徑處理、FxNLMS 訓練、FIR 轉 Biquad。
    
    Args:
        x (np.ndarray): 輸入訊號
        rew_file_path (str): REW 脈衝響應檔案路徑
        target_taps (int): 次級路徑 S(z) 的目標 tap 數量
        num_biquads (int): 預期擬合的 Biquad 數量
        mu (float): FxNLMS 學習率
        filter_length (int): FxNLMS 濾波器長度
    Examples:
        >>> x, d, fs, metadata = load_and_scale_dsp_dac_output_data("PDM_20_ref_7985.wav")
        >>> run_anc_simulation(x, "IR_R_Mar_25.txt", target_taps=512, num_biquads=4, mu=0.01, filter_length=256)
    """
    # 載入次級路徑 S(z) 並去除底噪
    S = load_rew_ir_and_denoise(rew_file_path, target_taps=2000, visualize=True, pre_peak_margin=10, taper_ratio=0.1)
    S_hat = load_rew_ir_and_denoise(rew_file_path, target_taps=target_taps, visualize=True, pre_peak_margin=10, taper_ratio=0.1)
    
    # 模擬初級路徑 P(z)
    P_simulated = create_simulated_primary_path(S, delay_samples=20, cutoff_freq=2000)
    d_n = np.convolve(x, P_simulated, mode='full')[:len(x)]
    
    # 執行 FxNLMS 訓練
    print("正在執行 FxNLMS 訓練...")
    F = np.zeros(target_taps)
    # 舊流程 (REW 的 S、模擬的 P)，保持原本的 leak / epsilon、不限幅
    e, y, w_fir, x_net = run_fxnlms(x, d_n, S, S_hat, F, mu=mu, filter_length=filter_length, leak=1e-5, eps=1e-6, clamp=None)
    
    # 繪製結果
    plot_time_domain_residual_comparison(x, e)
    
    '''
    # --- 步驟 4：將 FIR 權重轉換為 ADAU1787 的 Biquad 係數 ---
    print("啟動 DE + L-BFGS-B 混合優化器進行 IIR 轉換...")
    # 這裡呼叫我們前一階段寫好的 fit_iir_minimize 流程
    # (頻率點可設為 512 點，聚焦在 0 ~ 2000Hz 降噪有效區間進行擬合)
    freq_points, h_target = signal.freqz(w_fir, worN=512)
    sos_matrix = fit_iir_minimize(freq_points, h_target, num_biquads=num_biquads)
    
    # --- 步驟 5：輸出 SigmaStudio 格式 ---
    print_sigmastudio_coefficients(sos_matrix)
    '''

def plot_psd_comparison(d: np.ndarray, e: np.ndarray, fs: int = 48000, nfft: int = 8192, 
                        frequencies_to_mark: list = [],
                        visualize: bool = True):
    """
    計算並繪製原始誤差 d(n) 與殘留誤差 e(n) 的功率譜密度 (PSD)。
    用以觀察 FxNLMS / Biquad ANC 在特定頻率 (窄頻) 上的真實降噪深度。
    自動於指定頻率點 (frequencies_to_mark) 附近搜尋局部峰值並標註降噪量。
    
    Args:
        d (np.ndarray): ANC Off 時的誤差麥克風訊號 (原始噪音)
        e (np.ndarray): ANC On 且收斂後的殘留誤差
        fs (int): 取樣率
        nfft (int): FFT 點數 (建議 8192)
        frequencies_to_mark (list): 要標記的目標頻率點列表 (Hz)
        visualize (bool): 是否顯示圖形
    """
    # 確保訊號長度一致，並取最後一半的資料 (確保已達到穩態收斂)
    min_len = min(len(d), len(e))
    half_idx = min_len // 2
    d_steady = d[half_idx:min_len]
    e_steady = e[half_idx:min_len]

    # 使用 Welch 方法計算功率譜密度 (Power Spectral Density)
    f_d, pxx_d = signal.welch(d_steady, fs=fs, window='hann', nperseg=nfft)
    f_e, pxx_e = signal.welch(e_steady, fs=fs, window='hann', nperseg=nfft)

    # 轉換為 dB 刻度 (10 * log10)
    eps = 1e-12
    pxx_d_db = 10 * np.log10(pxx_d + eps)
    pxx_e_db = 10 * np.log10(pxx_e + eps)

    # 計算各頻率點的實際降噪深度 (Attenuation = d_dB - e_dB)
    attenuation = pxx_d_db - pxx_e_db

    # 搜尋指定頻率附近的局部峰值 (Local Peak) 並計算降噪量
    marked_info = []
    for freq in frequencies_to_mark:
        # 動態搜尋半徑 (預設 +-5% 或至少 +-25Hz)
        search_range = max(15.0, freq * 0.05)
        search_idx = np.where((f_d >= freq - search_range) & (f_d <= freq + search_range))[0]
        
        if len(search_idx) > 0:
            # 於原始噪音 d(n) 之 PSD 中找出局部能量峰值點
            peak_local_idx = np.argmax(pxx_d_db[search_idx])
            idx = search_idx[peak_local_idx]
        else:
            idx = np.argmin(np.abs(f_d - freq))
            
        actual_freq = f_d[idx]
        att_val = attenuation[idx]
        marked_info.append((freq, actual_freq, att_val))
        print(f"Target: {freq} Hz -> Local Peak: {actual_freq:.1f} Hz, Attenuation: {att_val:+.2f} dB")

    if visualize:
        # 繪圖
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.canvas.manager.set_window_title('PSD and Narrowband Attenuation')

        # 上半部：PSD 絕對能量比較
        ax1.semilogx(f_d, pxx_d_db, label='ANC OFF: Original $d(n)$', color='blue', alpha=0.7)
        ax1.semilogx(f_e, pxx_e_db, label='ANC ON: Residual $e(n)$', color='red', alpha=0.7)
        ax1.set_ylabel('Power Spectral Density (dB/Hz)')
        ax1.set_title('Power Spectral Density Comparison (Steady State)')
        ax1.legend(loc='upper right')
        ax1.grid(True, which="major", ls="-", alpha=0.6)
        ax1.grid(True, which="minor", ls="--", alpha=0.3)
        ax1.set_xlim([20, 20000])

        # 在上半部圖下方空白區加入文字標註 (頻率與降噪深度)
        if marked_info:
            text_lines = ["Narrowband Attenuation:"]
            for target_f, peak_f, att_v in marked_info:
                text_lines.append(f"  {peak_f:.1f} Hz: {att_v:+.2f} dB (Target: {target_f} Hz)")
            mark_text_box = "\n".join(text_lines)
            ax1.text(0.02, 0.05, mark_text_box, transform=ax1.transAxes,
                     fontsize=9, verticalalignment='bottom', horizontalalignment='left',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cccccc', alpha=0.85))

        # 下半部：各頻帶降噪深度 (維持原樣)
        # 將低於 0 dB (噪音放大) 的部分填色警告
        ax2.semilogx(f_d, attenuation, label='Attenuation (dB)', color='green')
        ax2.fill_between(f_d, attenuation, 0, where=(attenuation >= 0), color='green', alpha=0.3, label='Noise Reduced')
        ax2.fill_between(f_d, attenuation, 0, where=(attenuation < 0), color='red', alpha=0.3, label='Noise Enhanced')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Attenuation (dB)')
        ax2.set_title('Narrowband Attenuation Depth')
        ax2.legend(loc='upper right')
        ax2.grid(True, which="major", ls="-", alpha=0.6)
        ax2.grid(True, which="minor", ls="--", alpha=0.3)

        plt.tight_layout()
        plt.show()

    return f_d, attenuation, marked_info

class FIRFilter:
    """
    FIR 濾波器類別 (支援因果濾波、嚴格零相位濾波與波德圖繪製)
    """
    def __init__(
        self,
        cutoff: float | list = [100, 4000],
        pass_zero: Literal['bandpass', 'lowpass', 'highpass'] = 'bandpass',
        numtaps: int = 201,
        window: str = 'hamming',
        fs: int = 48000
    ):
        """
        初始化 FIR 濾波器

        Args:
            cutoff (float | list): 截止頻率 (Hz)，單一數值 (低通/高通) 或頻帶區間 [f_low, f_high] (帶通)
            pass_zero (Literal['bandpass', 'lowpass', 'highpass']): 通帶類型，對應 scipy.signal.firwin 的 pass_zero
            numtaps (int): 濾波器階數 / 長度 (若為偶數會自動 +1 確保整數群延遲)
            window (str): 窗函數類型 (預設 'hamming')
            fs (int): 取樣頻率 (Hz)
        """
        # 確保 numtaps 為奇數以獲得整數群延遲 (Type I FIR)
        if numtaps % 2 == 0:
            numtaps += 1

        self.cutoff = cutoff
        self.pass_zero = pass_zero
        self.numtaps = numtaps
        self.window = window
        self.fs = fs
        self.tau = (self.numtaps - 1) // 2  # 整數群延遲 (samples)

        # 預先設計並快取濾波器係數
        self.coeff = signal.firwin(
            numtaps=self.numtaps,
            cutoff=self.cutoff,
            pass_zero=self.pass_zero,
            window=self.window,
            fs=self.fs
        )

    def get_coeff(self) -> np.ndarray:
        """取得濾波器係數 (FIR taps)"""
        return self.coeff

    def get_tau(self) -> int:
        """取得群延遲點數 (samples)"""
        return self.tau

    def filter_causal(self, x: np.ndarray) -> np.ndarray:
        """
        標準因果卷積濾波 (訊號將被延遲 tau 點)
        """
        return signal.lfilter(self.coeff, 1.0, x)

    def filter_zero_phase(self, x: np.ndarray) -> np.ndarray:
        """
        使用奇數階對稱 FIR + 時間軸向左平移 tau 點，消除相位延遲達成零相位濾波
        """
        # 1. 因果卷積濾波
        y_causal = signal.lfilter(self.coeff, 1.0, x)
        # 2. 向左平移 tau 點消除群延遲
        y_zero_phase = np.roll(y_causal, -self.tau)
        # 3. 清除末端循環卷積產生的邊界偽影
        y_zero_phase[-self.tau:] = 0.0
        return y_zero_phase

    def __call__(self, x: np.ndarray, zero_phase: bool = True) -> np.ndarray:
        """
        支援物件直接呼叫進行濾波: y = fir(x)
        """
        return self.filter_zero_phase(x) if zero_phase else self.filter_causal(x)

    def plot_bode(self, title: str = "FIR Filter Bode Diagram"):
        """繪製波德圖 (Bode Diagram)"""
        plot_bode_diagram(self.coeff, fs=self.fs, title=title)

def resample_for_dsp(x: np.ndarray,
                     d: np.ndarray,
                     S_z: np.ndarray,
                     F_z: np.ndarray,
                     original_fs: int = 48000,
                     target_fs: int = 48000,
                     path_fs: int = 48000
                     ) -> tuple:
    """
    將錄音與路徑脈衝響應重取樣到 DSP 取樣率 (scipy.signal.resample_poly 多相濾波)。
    這裡不做帶通濾波：參考濾波器要放在 FxNLMS 迴路內 (run_fxnlms 的 ref_sos)，誤差訊號 d 不濾波。

    Args:
        x (np.ndarray): 參考麥克風訊號
        d (np.ndarray): 誤差麥克風訊號
        S_z (np.ndarray): 次級路徑脈衝響應
        F_z (np.ndarray): 回饋路徑脈衝響應
        original_fs (int): 錄音的原始取樣率
        target_fs (int): 目標取樣率 (DSP)
        path_fs (int): 路徑脈衝響應的取樣率

    Returns:
        tuple: (x_resample, d_resample, S_z_resample, F_z_resample)
            - x_resample (np.ndarray): 經過重取樣後的參考訊號
            - d_resample (np.ndarray): 經過重取樣後的誤差訊號
            - S_z_resample (np.ndarray): 經過重取樣後的次級路徑脈衝響應
            - F_z_resample (np.ndarray): 經過重取樣後的回饋路徑脈衝響應
    """
    x_resample = resample_data(x, original_fs=original_fs, target_fs=target_fs)
    d_resample = resample_data(d, original_fs=original_fs, target_fs=target_fs)
    F_z_resample = resample_data(F_z, original_fs=path_fs, target_fs=target_fs)
    S_z_resample = resample_data(S_z, original_fs=path_fs, target_fs=target_fs)

    return x_resample, d_resample, S_z_resample, F_z_resample

def compare_anc_with_and_without_filter(x: np.ndarray,
                                        d: np.ndarray,
                                        S: np.ndarray,
                                        S_hat: np.ndarray,
                                        F_hat: np.ndarray,
                                        ref_sos: np.ndarray,
                                        fs: int = 48000,
                                        eval_filter: FIRFilter = None,
                                        frequencies_to_mark: tuple = (600, 3600),
                                        lf_band: tuple = (100, 300),
                                        visualize: bool = True,
                                        mu_no_filter: float = None,
                                        **fxnlms_kwargs) -> dict:
    """
    比較「不濾波」與「參考訊號在迴路內濾波」兩種 FxNLMS，兩者都用未濾波的 d 與同一次模擬的 e 評估。

    濾波器只作用在送進 W 與 S_hat 的參考訊號 (run_fxnlms 的 ref_sos)，誤差麥克風的 d / e 不濾波，
    所以 ANC on 的 e 與 ANC off 的 d 可以直接相減比較頻譜。
    舊流程以零相位濾波 x、d 訓練 W，再凍結 W 套用到未濾波的 x：W 在帶外沒有被訓練，凍結後帶外會被放大，
    而且使用時拿掉了濾波器，結果不代表 DSP 實際執行的情況。

    Args:
        x (np.ndarray): 參考麥克風訊號 (已重取樣到 fs，未濾波)
        d (np.ndarray): 誤差麥克風訊號 (已重取樣到 fs，未濾波)
        S (np.ndarray): 次級路徑脈衝響應
        S_hat (np.ndarray): 次級路徑估測脈衝響應
        F_hat (np.ndarray): 回授路徑估測脈衝響應
        ref_sos (np.ndarray): 迴路內參考濾波器 (scipy sos 格式)
        fs (int): 取樣率
        eval_filter (FIRFilter): 選用，計算評估頻段降噪量的濾波器 (事後分析，可以用零相位)
        frequencies_to_mark (tuple): 要標示窄頻降噪量的頻率 (Hz)
        lf_band (tuple): 檢查低頻惡化的頻帶 (Hz)
        visualize (bool): 是否繪圖
        mu_no_filter (float): 不濾波那次的步長 (例如 FW_MU_NO_FILTER)；None 表示與濾波那次相同。
                              不濾波時低頻佔滿 x'，同樣的步長可能在低頻發散，所以兩者通常要用各自穩定的步長比較
        **fxnlms_kwargs: 傳給 run_fxnlms 的其餘參數 (mu、filter_length、leak 等)，兩次模擬相同；
                         例如 F=... 可模擬 F_hat 與真實回授路徑不同

    Returns:
        dict: {'no filter': {...}, 'in-loop filter': {...}}，各含 e, y, w, f, attenuation, marked,
              lf_worst_db, full_band_db, in_band_db
    """
    results = {}
    for label, sos in (('no filter', None), ('in-loop filter', ref_sos)):
        kwargs = dict(fxnlms_kwargs)
        if sos is None and mu_no_filter is not None:
            kwargs['mu'] = mu_no_filter
        print(f"\n=== {label}, mu = {kwargs.get('mu', FW_MU)} ===")
        e, y, w, _ = run_fxnlms(x, d, S, S_hat, F_hat, ref_sos=sos, **kwargs)
        f, attenuation, marked = plot_psd_comparison(d, e, fs=fs, nfft=8192,
                                                     frequencies_to_mark=list(frequencies_to_mark),
                                                     visualize=visualize)
        # 與 plot_psd_comparison 相同，取後半段 (已收斂) 計算
        half = min(len(d), len(e)) // 2
        full_band_db = 10 * np.log10(np.mean(d[half:]**2) / np.mean(e[half:]**2))
        lf = (f >= lf_band[0]) & (f <= lf_band[1])
        lf_worst_db = attenuation[lf].min()
        in_band_db = None
        if eval_filter is not None:
            d_in = eval_filter.filter_zero_phase(d[half:])
            e_in = eval_filter.filter_zero_phase(e[half:])
            in_band_db = 10 * np.log10(np.mean(d_in**2) / np.mean(e_in**2))
        print(f"全頻段降噪量: {full_band_db:+.2f} dB"
              + ("" if in_band_db is None else f", 評估頻段降噪量: {in_band_db:+.2f} dB")
              + f", {lf_band[0]}-{lf_band[1]} Hz 最差: {lf_worst_db:+.2f} dB")
        results[label] = dict(e=e, y=y, w=w, f=f, attenuation=attenuation, marked=marked,
                              lf_worst_db=lf_worst_db, full_band_db=full_band_db, in_band_db=in_band_db)
    return results

# ==========================================
# 工具一：時間軸截斷與加窗 (Time-Domain Windowing)
# ==========================================
def smooth_fir_by_windowing(w_fir: np.ndarray, target_taps: int = 256, taper_ratio: float = 0.2) -> np.ndarray:
    """
    將過長的 FIR 截短，並在尾端施加平滑衰減 (Fade-out)，去除晚期殘響雜訊。
    這會產生一組非常適合送給 Biquad 優化器去自動擬合的乾淨 FIR。
    在時域截斷等於在頻域捲積窗函數，會讓尖峰變成圓弧形。
    
    Args:
        w_fir: 原始 1024-tap FIR 權重
        target_taps: 想要保留的前段長度 (例如 256 或 128)
        taper_ratio: 尾端佔總截斷長度的衰減比例 (例如 0.2 代表最後 20% 會平滑降至 0)
    
    Returns:
        w_short: 處理後的短 FIR 陣列
    """
    if len(w_fir) <= target_taps:
        return w_fir
        
    # 1. 截取前段 (這裡包含了最核心的因果物理延遲與主降噪波)
    w_short = w_fir[:target_taps].copy()
    
    # 2. 計算衰減區間長度
    fade_len = int(target_taps * taper_ratio)
    
    if fade_len > 0:
        # 建立一個完整的 Hann 視窗，並只取它的後半段 (從 1 緩降到 0)
        hann_full = signal.windows.hann(fade_len * 2)
        fade_curve = hann_full[fade_len:]
        
        # 3. 將衰減曲線乘上 FIR 的尾巴，強迫隨機雜訊歸零
        w_short[-fade_len:] *= fade_curve
        
    return w_short

# ==========================================
# 工具二：分數八度音階頻譜平滑 (Fractional-Octave Smoothing)
# ==========================================
def fractional_octave_smoothing(freqs: np.ndarray, mag_db: np.ndarray, fraction: float = 1/3) -> np.ndarray:
    """
    對頻譜振幅進行 1/N 八度音階平滑。
    這非常適合用來畫圖，幫助您在 SigmaStudio 裡「手動」對齊 PEQ，而不會被毛刺干擾。
    
    Args:
        freqs: 頻率陣列 (Hz)
        mag_db: 對應的振幅陣列 (dB)
        fraction: 平滑度 (1/3 是業界標準，1/1 會非常圓滑，1/6 較保留細節)
        
    Returns:
        smoothed_mag: 平滑化後的振幅陣列 (dB)
    """
    smoothed_mag = np.zeros_like(mag_db)
    
    for i, fc in enumerate(freqs):
        if fc <= 0:
            smoothed_mag[i] = mag_db[i]
            continue
            
        # 計算這個中心頻率對應的「滑動視窗」上下界
        # 八度音階的特性是：高頻時視窗寬，低頻時視窗窄 (符合人類聽覺)
        f_min = fc * (2.0 ** (-fraction / 2.0))
        f_max = fc * (2.0 ** (fraction / 2.0))
        
        # 找出落在這個頻段內的所有資料點索引
        idx = (freqs >= f_min) & (freqs <= f_max)
        
        # 將這個頻段內的 dB 值取平均
        if np.any(idx):
            smoothed_mag[i] = np.mean(mag_db[idx])
        else:
            smoothed_mag[i] = mag_db[i]
            
    return smoothed_mag

# ==========================================
# 測試與視覺化示範
# ==========================================
def compare_fir_and_spectrum(w_fir: np.ndarray, fs: int = 48000, target_taps: int = 256, taper_ratio: float = 0.2):
    """
    將原始 FIR 與截斷加窗後的 FIR 做比較，並繪製頻譜響應。
    
    Args:
        w_fir: 原始 FIR 權重
        fs: 取樣率
        target_taps: 截斷後的 FIR 長度
        taper_ratio: 尾端平滑比例
    """
    w_fir_clean = smooth_fir_by_windowing(w_fir, target_taps=target_taps, taper_ratio=taper_ratio)
    
    # 計算頻譜
    eval_points = 1024
    w, h_raw = signal.freqz(w_fir, worN=eval_points, fs=fs)
    _, h_clean = signal.freqz(w_fir_clean, worN=eval_points, fs=fs)
    
    mag_raw = 20 * np.log10(np.abs(h_raw) + 1e-12)
    mag_clean = 20 * np.log10(np.abs(h_clean) + 1e-12)
    
    # 執行 1/3 八度音階平滑 (僅用於視覺化)
    mag_smoothed_1_3 = fractional_octave_smoothing(w, mag_raw, fraction=1/3)
    
    # 畫圖比較
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 上半部：時間軸波形比較
    ax1.plot(w_fir, label='Original FIR', alpha=0.5)
    ax1.plot(w_fir_clean, label='Truncated & Windowed FIR', linewidth=2)
    ax1.set_title("Time-Domain FIR Weights")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 下半部：頻譜響應比較
    valid_idx = (w >= 0) & (w <= 12000)
    ax2.plot(w[valid_idx], mag_raw[valid_idx], label='Raw Spectrum', alpha=0.3, color='gray')
    ax2.plot(w[valid_idx], mag_smoothed_1_3[valid_idx], label='1/3 Octave Smoothed', color='blue', linewidth=2)
    ax2.plot(w[valid_idx], mag_clean[valid_idx], label='Spectrum of Windowed FIR', color='red', linestyle='--', linewidth=2)

    ax2.set_title("Magnitude Response (Bode Plot)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def export_freq_response_for_rew(w_fir: np.ndarray, fs_target:int = 48000, filename:str = "Wz_response.txt"):
    """
    1. 剝離純延遲
    2. 生成 REW 響應檔
    """
    # 定位主能量峰值延遲
    delay_samples = np.argmax(np.abs(w_fir))
    
    # 頻響計算
    w, h = signal.freqz(w_fir, a=1, worN=4096, fs=fs_target)
    
    # 補償純延遲相位: H_comp(w) = H(w) * exp(j * w * delay / fs)
    phase_comp = np.exp(1j * 2 * np.pi * w * delay_samples / fs_target)
    h_min = h * phase_comp
    
    # 振幅取反 (dB)
    mag_db = 20 * np.log10(np.abs(h_min) + 1e-12)
    phase_deg = np.angle(h_min, deg=True)
    
    with open(filename, "w") as f:
        f.write("* W(z) for REW\n")
        f.write(f"* Extracted Delay: {delay_samples} samples at {fs_target} Hz\n")
        f.write("* Freq(Hz) Magnitude(dB) Phase(degrees)\n")
        for freq, mag, ph in zip(w, mag_db, phase_deg):
            f.write(f"{freq:.2f} {mag:.4f} {ph:.2f}\n")
            
    print(f"[Info] 檔案已輸出：{filename}")
    print(f"[Info] 請在 ADAU1787 FastDSP 前端配置 Delay = {delay_samples} samples")

def biquad_peaking(f0: float, gain_db: float, Q: float, fs: int = 384000):
    """依據 Audio EQ Cookbook 計算 384kHz 下的 Peaking EQ 係數"""
    A = 10**(gain_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A
    
    # 歸一化 a0
    return np.array([b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0])

def to_5_23_hex(val: float):
    """轉換浮點數為 ADAU1787 FastDSP 28-bit 5.23 格式十六進制字串"""
    # 飽和截斷在 [-16.0, 16.0 - 2^-23]
    val_clamped = np.clip(val, -16.0, 16.0 - (1.0 / (2**23)))
    int_val = int(np.round(val_clamped * (2**23)))
    if int_val < 0:
        int_val = (1 << 28) + int_val
    return f"0x{int_val & 0x0FFFFFFF:07X}"

def export_wav_for_rew(w_fir: np.ndarray, fs_target:int = 48000, filename:str = "Wz_response.wav"):
    """
    將 FIR 權重轉換為 WAV 檔，方便在 REW 中做時域與頻域分析。

    """
    # 確保資料型別為 float32 並去除異常值
    w_fir_clean = np.nan_to_num(w_fir.flatten()).astype(np.float32)
    
    # 輸出 WAV 檔案
    wavfile.write(filename, fs_target, w_fir_clean)
    print(f"[Info] FIR 權重已輸出為 WAV 檔：{filename}")

def parse_rew_text_and_convert(filepath, fs_target=384000, invert_gain=True, print_enabled=True):
    """
    解析 REW 導出的文字檔並計算 FastDSP 係數
    支援 CP950/UTF-8 編碼，僅讀取狀態為 ON 的濾波器
    """
    import re
    filters = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 匹配包含 PK (Peaking) 的有效濾波器列
            # 格式範例：Filter 1: ON  PK  Fc 600.0 Hz  Gain -12.5 dB  Q 8.00
            match = re.search(r'Filter\s+(\d+):\s+ON\s+PK\s+Fc\s+([\d\.]+)\s+Hz\s+Gain\s+([\-\d\.]+)\s+dB\s+Q\s+([\d\.]+)', line)
            if match:
                num = int(match.group(1))
                fc = float(match.group(2))
                gain_rew = float(match.group(3))
                q = float(match.group(4))
                # 若先前為正向消峰擬合，反轉符號以作為反相補償濾波器
                actual_gain = -gain_rew if invert_gain else gain_rew
                filters.append((num, fc, actual_gain, q))

    print(f"--- 成功解析 {len(filters)} 組 Biquad 濾波器 (目標取樣率: {fs_target} Hz) ---")
    sos_list = []
    for num, fc, g, q in filters:
        sos = biquad_peaking(fc, g, q, fs=fs_target)
        sos_list.append(sos)
        b0, b1, b2, _, a1, a2 = sos
        if print_enabled == True:
            print(f"// Filter {num}: Fc={fc} Hz, Actual Gain={g:+.2f} dB, Q={q}")
            print(f"B0:  {to_5_23_hex(b0)}  ({b0:+.8f})")
            print(f"B1:  {to_5_23_hex(b1)}  ({b1:+.8f})")
            print(f"B2:  {to_5_23_hex(b2)}  ({b2:+.8f})")
            print(f"-A1: {to_5_23_hex(-a1)}  ({-a1:+.8f})")
            print(f"-A2: {to_5_23_hex(-a2)}  ({-a2:+.8f})\n")
    return np.array(sos_list)

def plot_delay_sweep_analysis(delays: list, nr_band_list: list, nr_overall_list: list, best_delay: int, best_nr_band: float, fs: int = 48000, band_label: str = "100-4000Hz"):
    """
    繪製 Biquad 前置純延遲 (Delay Samples) 掃描與降噪量 (NR) 關聯分析圖
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(delays, nr_band_list, 'g.-', label=f'In-Band NR ({band_label})', linewidth=1.8, markersize=8)
    ax.plot(delays, nr_overall_list, 'b.--', label='Overall NR', alpha=0.7, linewidth=1.2, markersize=5)
    
    # 標記最佳延遲點
    best_time_ms = best_delay / fs * 1000.0
    ax.scatter([best_delay], [best_nr_band], color='red', s=60, zorder=5, 
               label=f'Optimal Delay: {best_delay} samples ({best_time_ms:.3f} ms) -> {best_nr_band:+.2f} dB')
    ax.axvline(x=best_delay, color='red', linestyle=':', alpha=0.6)
    
    ax.set_title(f'Biquad ANC Delay Sweep Analysis (Sampling Rate: {fs} Hz)\n[x(n) -> Delay z^(-D) -> Biquad SOS -> Secondary Path S(z)]')
    ax.set_xlabel('Delay (Samples)')
    ax.set_ylabel('Noise Reduction (dB)')
    ax.grid(True, which='both', ls=':', alpha=0.6)
    ax.legend(loc='best')
    
    # 增加上方 X 軸顯示對應時間 (ms)
    sec_ax = ax.secondary_xaxis('top', functions=(lambda s: s / fs * 1000.0, lambda ms: ms / 1000.0 * fs))
    sec_ax.set_xlabel('Delay Time (ms)')
    
    plt.tight_layout()
    plt.show()

def simulate_biquad_anc(x: np.ndarray, 
                        d: np.ndarray, 
                        sec_path: np.ndarray, 
                        sos_matrix: np.ndarray, 
                        fs: int = 48000, 
                        delay_samples: int | list | range | np.ndarray = None,
                        eval_filter: FIRFilter = None,
                        eval_cutoff: float | list = [100, 4000],
                        frequencies_to_mark: list = [],
                        visualize: bool = True) -> tuple:
    """
    模擬 Biquad 串聯 ANC 系統，測試並確認最佳的延遲樣本數 (Delay Samples)。

    【物理意義與訊號鏈架構】:
    - REW 透過等化器擬合出的 Biquad PEQ (sos_matrix) 屬於最小相位 (Minimum Phase) 濾波器，
      其剝離了聲學傳播與硬體的純延遲 (Pure Transport Delay)。
    - 在實際 DSP 硬體 (如 ADAU1787 FastDSP) 與模擬中，必須在 sos_matrix 前方串聯一個純延遲模組 z^(-D):
      x(n) ---> [ Delay z^(-D) ] ---> [ Biquad PEQ: W_min(z) ] ---> y(n) ---> [ S(z) ] ---> y_sec(n)
      殘差: e(n) = d(n) - y_sec(n) (或同相 + y_sec(n))
    - 本函式可輸入單一延遲測試，亦可傳入範圍 (如 range(0, 50) 或 None) 自動掃描尋求降噪深度最大之最佳 delay。

    Args:
        x (np.ndarray): 參考麥克風訊號 x(n)
        d (np.ndarray): 誤差麥克風原始噪音訊號 d(n)
        sec_path (np.ndarray): 次級路徑脈衝響應 S(z)
        sos_matrix (np.ndarray): REW 產生的最小相位 Biquad SOS 矩陣
        fs (int): 取樣頻率 (Hz)
        delay_samples (int | list | range | np.ndarray | None):
            - 若為單一 int: 直接以此固定 delay 進行閉迴路模擬。
            - 若為 list / range: 掃描測試範圍內所有 delay，找出最佳配置。
            - 若為 None: 自動掃描預設範圍 range(0, 64)。
        eval_filter (FIRFilter | None): 用於評估帶內降噪量的零相位 FIRFilter 物件；若為 None 則自動依 eval_cutoff 建立。
        eval_cutoff (float | list): 帶內評估頻率範圍 (Hz)，例如 [300, 4000] 或 [100, 4000]。
        frequencies_to_mark (list): 在 PSD 圖上重點標記的頻率點 (Hz)
        visualize (bool): 是否顯示圖形 (包含 Delay-NR 掃描圖、PSD 頻譜圖與時域殘差波形)

    Returns:
        tuple: (best_delay, best_nr_band, best_nr_overall, e_best)
            - best_delay (int): 降噪效果最佳的延遲點數 (samples)
            - best_nr_band (float): 最佳有效帶限降噪量 (dB)
            - best_nr_overall (float): 最佳全頻段降噪量 (dB)
            - e_best (np.ndarray): 最佳延遲配置下對應的殘差訊號 e(n)
    """
    # 確保資料為 1D 浮點數且長度一致
    x = np.nan_to_num(x.flatten()).astype(np.float64)
    d = np.nan_to_num(d.flatten()).astype(np.float64)
    sec_path = np.nan_to_num(sec_path.flatten()).astype(np.float64)
    min_len = min(len(x), len(d))
    x, d = x[:min_len], d[:min_len]

    # 解析 delay_samples 輸入參數
    if delay_samples is None:
        delay_list = list(range(0, 64))
        is_sweep = True
    elif isinstance(delay_samples, (list, range, np.ndarray, tuple)):
        delay_list = list(delay_samples)
        is_sweep = len(delay_list) > 1
    elif isinstance(delay_samples, (int, np.integer)):
        delay_list = [int(delay_samples)]
        is_sweep = False
    else:
        raise ValueError(f"不支援的 delay_samples 型別: {type(delay_samples)}")

    # ==========================================
    # 初始化評估用帶內濾波器 (採用零相位 FIR 以獲得最高精度無相位失真)
    # ==========================================
    if eval_filter is None:
        pass_zero = 'bandpass' if isinstance(eval_cutoff, (list, tuple)) and len(eval_cutoff) == 2 else 'lowpass'
        eval_filter = FIRFilter(cutoff=eval_cutoff, pass_zero=pass_zero, numtaps=201, fs=fs)

    if isinstance(eval_filter.cutoff, (list, tuple)):
        band_str = f"{eval_filter.cutoff[0]}-{eval_filter.cutoff[1]}Hz"
    else:
        band_str = f"{eval_filter.cutoff}Hz"

    # ==========================================
    # 執行閉迴路訊號鏈過濾 (利用 LTI 特性預先計算基底響應以大幅提速)
    # 訊號鏈: x(n) -> Delay(D) -> Biquad(sos) -> S(z) 等價於 [x(n) -> Biquad(sos) -> S(z)] -> Delay(D)
    # ==========================================
    y_base = signal.sosfilt(sos_matrix, x)
    y_sec_base = signal.lfilter(sec_path, 1, y_base)

    # 略過前 0.2 秒暫態建立期
    eval_start = int(0.2 * fs)
    d_bp = eval_filter.filter_zero_phase(d)
    p_d = np.mean(d[eval_start:]**2)
    p_d_bp = np.mean(d_bp[eval_start:]**2)

    sweep_results = []

    for D in delay_list:
        # 在前端串聯純延遲: x_del(n) = x(n - D)
        y_sec_del = np.zeros_like(y_sec_base)
        if D == 0:
            y_sec_del = y_sec_base
        elif D < len(y_sec_base):
            y_sec_del[D:] = y_sec_base[:-D]

        # 殘差計算 (自動偵測反相抵消或同相疊加)
        e_opt1 = d - y_sec_del
        e_opt2 = d + y_sec_del
        p_e1 = np.mean(e_opt1[eval_start:]**2)
        p_e2 = np.mean(e_opt2[eval_start:]**2)

        if p_e1 <= p_e2:
            e_cand = e_opt1
            pol = "d(n) - y_sec(n) [標準反相]"
            p_e = p_e1
        else:
            e_cand = e_opt2
            pol = "d(n) + y_sec(n) [同相疊加]"
            p_e = p_e2

        nr_overall = 10 * np.log10(p_d / (p_e + 1e-12))

        # 使用零相位 FIR 計算帶限降噪量
        e_cand_bp = eval_filter.filter_zero_phase(e_cand)
        p_e_bp = np.mean(e_cand_bp[eval_start:]**2)
        nr_band = 10 * np.log10(p_d_bp / (p_e_bp + 1e-12))

        sweep_results.append({
            'delay': D,
            'nr_band': nr_band,
            'nr_overall': nr_overall,
            'polarity': pol,
            'e': e_cand
        })

    # 尋找最佳延遲點 (以目標頻段帶限降噪深度為主要依據)
    best_result = max(sweep_results, key=lambda res: res['nr_band'])
    best_delay = best_result['delay']
    best_nr_band = best_result['nr_band']
    best_nr_overall = best_result['nr_overall']
    e_best = best_result['e']
    best_polarity = best_result['polarity']

    # ==========================================
    # 輸出分析報告
    # ==========================================
    if is_sweep:
        print("\n=========================================================================")
        print(f"--- Biquad ANC 前置 Delay 掃描分析 (共測試 {len(delay_list)} 組延遲配置) ---")
        print(f"【評估頻段】:     {band_str} (使用零相位 FIR 濾波器)")
        print(f"【最佳前置延遲】: {best_delay} samples ({best_delay/fs*1000.0:.3f} ms / {best_delay/fs*1e6:.1f} us)")
        print(f"極性判定方式:     {best_polarity}")
        print(f"最佳有效帶限降噪: {best_nr_band:+.2f} dB ({band_str})")
        print(f"最佳全頻段降噪:   {best_nr_overall:+.2f} dB")
        
        # 若有包含 Delay=0，計算相比於無延遲 (Delay=0) 帶來的性能提升
        zero_res = next((item for item in sweep_results if item['delay'] == 0), None)
        if zero_res is not None and best_delay != 0:
            diff_band = best_nr_band - zero_res['nr_band']
            print(f"相較於未加 Delay (0 sample): 帶限降噪改善了 {diff_band:+.2f} dB (從 {zero_res['nr_band']:+.2f} dB 提升至 {best_nr_band:+.2f} dB)")
        
        print("-------------------------------------------------------------------------")
        print("Top 5 最佳 Delay 候選排行:")
        top5 = sorted(sweep_results, key=lambda res: res['nr_band'], reverse=True)[:min(5, len(sweep_results))]
        for rank, item in enumerate(top5, 1):
            print(f"  #{rank}: Delay={item['delay']:2d} samples ({item['delay']/fs*1000.0:6.3f} ms) | 帶限 NR: {item['nr_band']:+6.2f} dB | 全頻 NR: {item['nr_overall']:+6.2f} dB")
        print("=========================================================================\n")
    else:
        print("\n==========================================")
        print(f"前置延遲配置: {best_delay} samples ({best_delay/fs*1000.0:.3f} ms / {best_delay/fs*1e6:.1f} us)")
        print(f"極性判定方式: {best_polarity}")
        print(f"全頻段降噪量 (Overall NR): {best_nr_overall:+.2f} dB")
        print(f"有效帶限降噪量 ({band_str}): {best_nr_band:+.2f} dB")
        print("==========================================\n")

    # ==========================================
    # 圖形視覺化
    # ==========================================
    if is_sweep and visualize:
        delays = [item['delay'] for item in sweep_results]
        nr_band_list = [item['nr_band'] for item in sweep_results]
        nr_overall_list = [item['nr_overall'] for item in sweep_results]
        plot_delay_sweep_analysis(delays, nr_band_list, nr_overall_list, best_delay, best_nr_band, fs=fs, band_label=band_str)

    if visualize:
        plot_psd_comparison(d, e_best, fs=fs, frequencies_to_mark=frequencies_to_mark, visualize=visualize)
        plot_time_domain_residual_comparison(d, e_best, fs=fs)

    return best_delay, best_nr_band, best_nr_overall, e_best

def plot_bode_diagram(filter_data: np.ndarray|tuple, fs=48000, title="Filter Bode Diagram"):
    """
    計算並繪製濾波器之波德圖 (振幅、解卷繞相位、群延遲)
    使用數值微分計算 Group Delay
    
    通用波德圖繪製函式:
    - 若 filter_data 為 1D 陣列: 自動識別為 FIR 係數 (fir_coeff)
    - 若 filter_data 為 2D 陣列 (N, 6): 自動識別為 IIR SOS 矩陣
    - 若 filter_data 為 tuple (b, a): 自動識別為傳遞函數多項式
    """
    # 1. 自動辨識輸入格式並計算頻率響應
    if isinstance(filter_data, np.ndarray) and filter_data.ndim == 1:
        # FIR 係數
        w, h = signal.freqz(filter_data, a=1.0, worN=8192, fs=fs)
    elif isinstance(filter_data, np.ndarray) and filter_data.ndim == 2:
        # IIR SOS 矩陣
        w, h = signal.sosfreqz(filter_data, worN=8192, fs=fs)
    elif isinstance(filter_data, tuple):
        # (b, a)
        b, a = filter_data
        w, h = signal.freqz(b, a, worN=8192, fs=fs)
    else:
        raise ValueError("不支援的輸入格式。")
    
    # 2. 計算振幅與解卷繞相位
    mag_db = 20 * np.log10(np.abs(h) + 1e-12)
    # 解卷繞相位 (Unwrapped Phase)
    phase_rad = np.unwrap(np.angle(h))
    phase_deg = phase_rad * 180.0 / np.pi
    # 3. 數值微分計算群延遲: tau_g = - d(phase_rad) / d(omega)
    # omega = 2 * pi * f (rad/s)
    d_omega = 2.0 * np.pi * (w - w[0])
    gd_seconds = -np.gradient(phase_rad, d_omega)
    gd_ms = gd_seconds * 1000.0  # 轉換為毫秒 (ms)
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # 振幅響應
    axs[0].plot(w, mag_db, 'b', lw=1.5)
    axs[0].set_ylabel('Magnitude (dB)')
    axs[0].set_title(title)
    axs[0].grid(True, which='both', ls=':')
    
    # 相位響應
    axs[1].plot(w, phase_deg, 'r', lw=1.5)
    axs[1].set_ylabel('Unwrapped Phase (deg)')
    axs[1].grid(True, which='both', ls=':')
    
    # 群延遲
    axs[2].plot(w, gd_ms, 'g', lw=1.5)
    axs[2].set_ylabel('Group Delay (ms)')
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_xlim([20, 10000])
    axs[2].grid(True, which='both', ls=':')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dsp_fs = FW_FS
    record_name = "100_dual_36159_60s.npz"
    x, d, fs, metadata = load_and_scale_dsp_dac_output_data(record_name)   # 板子單位 (ADAU1979 滿刻度)
    # 板子經 48 kHz 核心邊界量到的 S / F (韌體 repo 的 second_path.h / feedback_path.h)
    # 錄音時喇叭靜音，x 不含回授；run_fxnlms 會把喇叭回授 F 加回參考麥克風，再用 F_hat 扣掉
    S_z, path_fs = load_anc_path_header(os.path.join(FIRMWARE_DIR, 'second_path.h'))
    F_z, _ = load_anc_path_header(os.path.join(FIRMWARE_DIR, 'feedback_path.h'))
    x_resample, d_resample, S_z_resample, F_z_resample = resample_for_dsp(x, d, S_z, F_z, original_fs=fs, target_fs=dsp_fs, path_fs=path_fs)
    # 比較不濾波與迴路內參考帶通 (因果 IIR)，兩者都用未濾波的 d / e 評估
    # 其餘參數用韌體預設值 (FW_*)；限幅放到 DAC 滿刻度 1.0，先比較濾波器位置本身的效果
    # (韌體的 FW_OUTPUT_LIMIT = 0.2 在 100% duty 會一直截斷：600 Hz 約需 0.2、3.6 kHz 約需 0.7)
    # 不濾波時用 FW_MU_NO_FILTER：用 FW_MU 會在低頻發散 (100-300 Hz 惡化約 -47 dB)
    ref_sos = signal.butter(4, FW_REF_BAND, 'bandpass', fs=dsp_fs, output='sos')
    compare_anc_with_and_without_filter(x_resample, d_resample, S_z_resample, S_z_resample, F_z_resample, ref_sos, fs=dsp_fs,
                                        eval_filter=FIRFilter(cutoff=[300, 4000], fs=dsp_fs), clamp=1.0,
                                        mu_no_filter=FW_MU_NO_FILTER)
    # 舊的 ADAU1787 Biquad 流程 (8 月的 REW S，SC598 已不使用):
    # S_rew = load_rew_ir_and_denoise("R Aug 3 secondary path.txt", target_taps=2048, pre_peak_margin=20, taper_ratio=0.1, visualize=False)
    # sos_matrix = parse_rew_text_and_convert("REW_500hz_4000hz_biquad_100_duty.txt", fs_target=dsp_fs, invert_gain=True, print_enabled=False)
    # simulate_biquad_anc(x_resample, d_resample, S_rew, sos_matrix, fs=dsp_fs, delay_samples=range(0, 75), eval_cutoff=[300, 4000], frequencies_to_mark=[600, 3600])