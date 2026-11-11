import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, Coupling
import scipy.io.wavfile as wavfile
import os
import re
import datetime

# 錄音存到這個 repo 的 measurements/YYYY-MM-DD/ (gitignored)，和用它的 FxNLMS.py 放在一起
MEASUREMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'measurements')

def acquire_ni9234_dual_channel(
    device_name: str, 
    ch0: str = "ai0", 
    ch1: str = "ai1", 
    fs: int = 48000, 
    duration_sec: float = 5.0
) -> tuple:
    """
    自 NI-9234 同步擷取雙通道電壓數據。
    
    Args:
        device_name (str): NI DAQ 裝置名稱 (如 'cDAQ1Mod1' 或 'Dev1'，需於 NI MAX 中確認)。
        ch0 (str): 第一通道名稱 (對應 Reference Mic)。
        ch1 (str): 第二通道名稱 (對應 Error Mic)。
        fs (int): 目標取樣頻率 (Hz)。注意：NI-9234 具備離散的硬體取樣率，DAQmx 會自動匹配至最接近的支援頻率。
        duration_sec (float): 擷取時間長度 (秒)。
        
    Returns:
        tuple: (x_data, d_data, actual_fs)
            - x_data (np.ndarray): ch0 的電壓數據 (Volts)，浮點數陣列。
            - d_data (np.ndarray): ch1 的電壓數據 (Volts)，浮點數陣列。
            - actual_fs (float): 硬體實際執行的取樣頻率。
    """
    samples_per_channel = int(fs * duration_sec)
    channel_list = [f"{device_name}/{ch0}", f"{device_name}/{ch1}"]
    
    with nidaqmx.Task() as task:
        # 1. 建立類比輸入電壓通道
        for chan in channel_list:
            ai_channel = task.ai_channels.add_ai_voltage_chan(
                physical_channel=chan,
                terminal_config=TerminalConfiguration.PSEUDO_DIFF,
                min_val=-5.0,  # NI-9234 量程通常為 +/- 5V
                max_val=5.0
            )
            
            # 2. 硬體保護與訊號設定 (針對非 IEPE 主動音訊設備)
            # 關閉 IEPE 激磁電流 (0 mA) 避免損毀連接設備
            try:
                ai_channel.ai_excit_val = 0.0
            except nidaqmx.errors.DaqError:
                pass # 部分驅動版本或模組於電壓模式下預設關閉，若無法寫入則忽略
                
            # 設定為交流耦合，濾除 ADAU1787 DAC 可能帶有的直流偏壓
            ai_channel.ai_coupling = Coupling.AC

        # 3. 設定取樣時鐘
        task.timing.cfg_samp_clk_timing(
            rate=fs,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=samples_per_channel
        )
        
        # 取得硬體實際設定之取樣頻率 (NI-9234 受限於內部 Delta-Sigma ADC 時脈除頻)
        actual_fs = task.timing.samp_clk_rate
        print(f"要求取樣率: {fs} Hz, 硬體實際取樣率: {actual_fs} Hz")
        samples_per_channel = int(actual_fs * duration_sec)
        # 再次配置
        task.timing.cfg_samp_clk_timing(
            rate=actual_fs,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=samples_per_channel
        )

        # 4. 執行擷取
        # 設定 timeout 需大於擷取總時間，避免提前逾時
        timeout = duration_sec + 2.0
        print(f"開始擷取 {duration_sec} 秒數據...")
        
        # read_all_avail_samp=False 確保讀取指定數量的樣本
        data = task.read(number_of_samples_per_channel=samples_per_channel, timeout=timeout)
        
        print("擷取完成。")
        
    # data 的結構為 list of lists，將其轉換為 numpy 陣列
    x_data = np.array(data[0])
    d_data = np.array(data[1])
    
    return x_data, d_data, actual_fs

def save_acquisition_data(filename: str, x_data: np.ndarray, d_data: np.ndarray, fs: float, format: str = 'wav', **meta):
    """
    將雙通道擷取數據儲存至硬碟。
    
    Args:
        filename (str): 檔案名稱 (不含副檔名或含副檔名皆可)。
        x_data (np.ndarray): 參考麥克風電壓陣列。
        d_data (np.ndarray): 誤差麥克風電壓陣列。
        fs (float): 實際取樣頻率。
        format (str): 'wav' 或 'npz'。
        **meta: 一併存進 npz 的量測條件，例如 duty=100, rpm=36159 (wav 不存)。
    """
    # 只去掉 .wav / .npz：名稱裡的小數點不是副檔名 (例如 50_dual_17960_2.5s)
    base_name, ext = os.path.splitext(filename)
    if ext.lower() not in ('.wav', '.npz'):
        base_name = filename
    
    if format.lower() == 'wav':
        # 合併為雙通道陣列，形狀為 (samples, channels)
        stereo_data = np.column_stack((x_data, d_data))
        # 轉換為 32-bit 浮點數，確保儲存電壓值不失真
        stereo_data_f32 = stereo_data.astype(np.float32)
        
        out_name = f"{base_name}.wav"
        # 依據 IEEE 754 標準，scipy 可直接寫入 float32
        wavfile.write(out_name, int(fs), stereo_data_f32)
        print(f"數據已儲存為 32-bit Float WAV: {out_name}")
        
    elif format.lower() == 'npz':
        out_name = f"{base_name}.npz"
        # 儲存為未壓縮的 NumPy 二進位檔案，保留 64-bit 精度
        np.savez(out_name, x=x_data, d=d_data, fs=fs, **meta)
        print(f"數據已儲存為 NumPy 陣列: {out_name}")
        
    else:
        raise ValueError("不支援的格式。請選擇 'wav' 或 'npz'。")

def record_name(duty: int, rpm: int, duration_sec: float, tag: str = '') -> str:
    """
    由量測條件組出錄音檔名 <duty>_dual_<rpm>_<秒數>s[_<標籤>] (不含副檔名)，
    例如 (100, 36159, 60.0) -> 100_dual_36159_60s，(100, 36159, 10, 'refA') -> 100_dual_36159_10s_refA。

    用途：FxNLMS.py 的 fan_tones() 用這個轉速決定評估時要量哪一條線 (轉速音 f0 與 BPF)，
    轉速填錄音當下的讀數即可，與實際值差 5% 以內都找得到。檔名由程式組成，不必手打整串名稱。
    標籤標示 mic 位置等條件：同一天同樣 duty、轉速的多次錄音靠它區分 (例如調 mic 位置找相干性時)。

    Raises:
        ValueError: duty 不在 1-100、轉速不大於 0 (與 fan_tones() 的檔名檢驗相同)，或標籤有英數字和 - 以外的字元
    """
    if not (1 <= duty <= 100) or rpm <= 0:
        raise ValueError(f"duty ({duty}) 須在 1-100、轉速 ({rpm}) 須大於 0")
    if tag and not re.fullmatch(r'[A-Za-z0-9-]+', tag):
        raise ValueError(f"標籤 ({tag}) 只能用英數字和 -")
    return f"{duty}_dual_{rpm}_{duration_sec:g}s" + (f"_{tag}" if tag else "")

def run_daq(device_name: str, duty: int, rpm: int, fs: int = 48000, duration_sec: float = 5.0, tag: str = '',
            out_dir: str = None):
    """
    執行 NI-9234 雙通道擷取並儲存數據 (.wav 與 .npz)。

    存到 out_dir，預設 MEASUREMENTS_DIR/<今天日期 YYYY-MM-DD>/。檔名由 record_name() 依 duty、轉速、秒數與標籤組成；
    npz 另存 duty、rpm、tag 與錄音開始時間 recorded，檔案改名或搬走也不會遺失。
    已有同名檔案時不擷取，避免覆蓋舊的錄音。

    Args:
        device_name (str): NI DAQ 裝置名稱。
        duty (int): 風扇 duty (%)，1-100。
        rpm (int): 錄音當下的風扇轉速讀數 (rpm)。
        fs (int): 取樣頻率。
        duration_sec (float): 擷取持續時間。
        tag (str): 標籤 (mic 位置等)，英數字和 -，可空白。
        out_dir (str): 存檔資料夾；None 時用 MEASUREMENTS_DIR/<今天日期>/。
    Example:
        run_daq(device_name="cDAQ1Mod1", duty=100, rpm=36159, fs=48000, duration_sec=60.0, tag="refA")
    """
    if out_dir is None:
        out_dir = os.path.join(MEASUREMENTS_DIR, datetime.date.today().isoformat())
    name = record_name(duty, rpm, duration_sec, tag)
    filename = os.path.join(out_dir, name)
    existing = [f"{name}{ext}" for ext in ('.wav', '.npz') if os.path.exists(f"{filename}{ext}")]
    if existing:
        print(f"{out_dir} 已有同名檔案 {', '.join(existing)}，不擷取以免覆蓋。請確認 duty 與轉速讀數，或換一個標籤。")
        return

    try:
        recorded = datetime.datetime.now().isoformat(timespec='seconds')
        x_rec, d_rec, actual_fs = acquire_ni9234_dual_channel(
            device_name=device_name,
            fs=fs,
            duration_sec=duration_sec
        )

        print(f"x 陣列形狀: {x_rec.shape}, 數值範圍: [{x_rec.min():.4f}, {x_rec.max():.4f}] V")
        print(f"d 陣列形狀: {d_rec.shape}, 數值範圍: [{d_rec.min():.4f}, {d_rec.max():.4f}] V")

        # 儲存數據
        os.makedirs(out_dir, exist_ok=True)
        save_acquisition_data(filename, x_rec, d_rec, actual_fs, format='wav')
        save_acquisition_data(filename, x_rec, d_rec, actual_fs, format='npz', duty=duty, rpm=rpm, tag=tag,
                              recorded=recorded)

        # 繪製前 1000 個取樣點驗證
        import matplotlib.pyplot as plt
        t = np.arange(1000) / actual_fs
        plt.figure(figsize=(10, 4))
        plt.plot(t, x_rec[:1000], label='Reference (ch0)')
        plt.plot(t, d_rec[:1000], label='Error (ch1)', alpha=0.7)
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except nidaqmx.errors.DaqError as e:
        print(f"DAQ 硬體錯誤: {e}")
        print("請確認 NI-DAQmx 驅動程式已安裝，且裝置名稱與 NI MAX 中顯示一致。")

if __name__ == "__main__":
    # 每次錄音都重新輸入 duty、轉速讀數與標籤，檔名由程式組成，不會沿用上一段的轉速
    # 存到 MEASUREMENTS_DIR/<今天日期>/ (這個 repo 的 measurements/)
    duration_sec = 60.0
    try:
        duty = int(input("風扇 duty (%): "))
        rpm = int(input("風扇轉速讀數 (rpm): "))
        tag = input("標籤 (mic 位置等，英數字和 -，可空白): ").strip()
        name = record_name(duty, rpm, duration_sec, tag)
        print(f"存到 {os.path.join(MEASUREMENTS_DIR, datetime.date.today().isoformat(), name)}")
    except ValueError as e:
        print(f"輸入無效: {e}")
    else:
        run_daq(device_name="cDAQ1Mod1", duty=duty, rpm=rpm, fs=48000, duration_sec=duration_sec, tag=tag)