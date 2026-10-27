import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, find_peaks, spectrogram, welch

def traditional_ecg_pipeline_demo():
    # ==========================================
    # 1. SIGNAL ACQUISITION (Synthetic Generation)
    # ==========================================
    fs = 500  # Sampling frequency (Hz)
    duration = 10  # seconds
    t = np.linspace(0, duration, fs * duration, endpoint=False)
    
    # Create a single ECG beat template (Gaussian sum)
    beat_len = int(fs * 60 / 75)  # 75 BPM -> 0.8s per beat -> 400 samples
    template = np.zeros(beat_len)
    template[100:140] = 0.15 * np.exp(-((np.arange(40)-20)**2)/(2*5**2))
    template[150:190] = -0.2 * np.exp(-((np.arange(40)-10)**2)/(2*3**2))
    template[190:210] = 1.0 * np.exp(-((np.arange(20)-10)**2)/(2*2**2))
    template[210:240] = -0.25 * np.exp(-((np.arange(30)-15)**2)/(2*4**2))
    template[280:340] = 0.3 * np.exp(-((np.arange(60)-30)**2)/(2*10**2))
    
    n_beats = int(np.ceil(len(t) / beat_len)) 
    ecg_clean = np.tile(template, n_beats)[:len(t)]
    
    # Add Real-World Noise
    baseline_wander = 0.8 * np.sin(2 * np.pi * 0.15 * t)
    powerline_noise = 0.4 * np.sin(2 * np.pi * 50 * t)
    muscle_noise = 0.15 * np.random.randn(len(t))
    
    ecg_noisy = ecg_clean + baseline_wander + powerline_noise + muscle_noise
    
    # ==========================================
    # 2. PRE-PROCESSING
    # ==========================================
    notch_sos = butter(2, [49, 51], btype='bandstop', fs=fs, output='sos')
    ecg_no_powerline = sosfilt(notch_sos, ecg_noisy) 
    
    bp_sos = butter(2, [0.5, 40], btype='band', fs=fs, output='sos')
    ecg_filtered = sosfilt(bp_sos, ecg_no_powerline)
    
    # ==========================================
    # 3. PHYSIOLOGICAL FEATURE EXTRACTION (R-Peaks & HRV)
    # ==========================================
    peaks, _ = find_peaks(ecg_filtered, distance=fs*0.5, prominence=0.5)
    rr_intervals = np.diff(peaks) / fs  # in seconds
    heart_rate = 60 / np.mean(rr_intervals)
    sdnn = np.std(rr_intervals) * 1000  # Standard Deviation of NN intervals (ms)
    
    # ==========================================
    # 4. TIME & FREQUENCY DOMAIN TRANSFORMATIONS
    # ==========================================
    # Time-Frequency (Spectrogram)
    f_spec, t_spec, Sxx = spectrogram(ecg_filtered, fs=fs, nperseg=128, noverlap=64)
    
    # Frequency Domain (Power Spectral Density via Welch's method)
    freqs, psd = welch(ecg_filtered, fs=fs, nperseg=256)
    
    # ==========================================
    # 5. ADVANCED FEATURE CALCULATION
    # ==========================================
    # Time-Domain Features
    mean_amp = np.mean(ecg_filtered)
    std_amp = np.std(ecg_filtered)
    # Zero Crossing Rate (ZCR)
    zcr = np.sum(np.abs(np.diff(np.signbit(ecg_filtered)))) / len(ecg_filtered)
    
    # Frequency-Domain Features (from PSD)
    # Spectral Centroid (Center of mass of the spectrum)
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
    # Spectral Bandwidth (Spread of the spectrum around the centroid)
    spectral_bandwidth = np.sqrt(np.sum(psd * (freqs - spectral_centroid)**2) / np.sum(psd))
    # Spectral Entropy (Measure of signal complexity/flatness)
    psd_norm = psd / np.sum(psd)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    
    # Console Output
    print("="*50)
    print("TRADITIONAL PIPELINE: EXTRACTED FEATURES")
    print("="*50)
    print("--- Physiological Features ---")
    print(f"Mean Heart Rate:      {heart_rate:.2f} BPM")
    print(f"SDNN (HRV metric):    {sdnn:.2f} ms")
    print("\n--- Time-Domain Features ---")
    print(f"Mean Amplitude:       {mean_amp:.4f} mV")
    print(f"Standard Deviation:   {std_amp:.4f} mV")
    print(f"Zero Crossing Rate:   {zcr:.4f}")
    print("\n--- Frequency-Domain Features ---")
    print(f"Spectral Centroid:    {spectral_centroid:.2f} Hz")
    print(f"Spectral Bandwidth:   {spectral_bandwidth:.2f} Hz")
    print(f"Spectral Entropy:     {spectral_entropy:.4f} bits")
    print("="*50)
    
    # ==========================================
    # VISUALIZATION
    # ==========================================
    fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=False)
    
    # Plot 1: Raw Noisy Signal
    axs[0].plot(t, ecg_noisy, color='gray', linewidth=0.8)
    axs[0].set_title('1. Raw Acquisition: Noisy ECG', fontsize=12, fontweight='bold')
    axs[0].set_ylabel('Amplitude (mV)')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Filtered Signal
    axs[1].plot(t, ecg_filtered, color='blue', linewidth=0.8)
    axs[1].set_title('2. Pre-processing: Bandpass & Notch Filtered', fontsize=12, fontweight='bold')
    axs[1].set_ylabel('Amplitude (mV)')
    axs[1].grid(True, linestyle='--', alpha=0.6)
    
    # Plot 3: Feature Extraction (R-Peaks)
    axs[2].plot(t, ecg_filtered, color='blue', linewidth=0.8)
    axs[2].plot(t[peaks], ecg_filtered[peaks], "x", color='red', markersize=10, label=f'R-peaks (HR: {heart_rate:.1f} BPM)')
    axs[2].set_title('3. Physiological Feature: R-Peak Detection', fontsize=12, fontweight='bold')
    axs[2].set_ylabel('Amplitude (mV)')
    axs[2].legend(loc='upper right')
    axs[2].grid(True, linestyle='--', alpha=0.6)
    
    # Plot 4: Spectrogram
    axs[3].pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='jet')
    axs[3].set_title('4. Transformation: Time-Frequency (Spectrogram)', fontsize=12, fontweight='bold')
    axs[3].set_ylabel('Frequency (Hz)')
    axs[3].set_xlabel('Time (s)')
    
    # Plot 5: Power Spectral Density (Welch)
    axs[4].plot(freqs, psd, color='purple', linewidth=1.5)
    axs[4].axvline(spectral_centroid, color='red', linestyle='--', label=f'Centroid: {spectral_centroid:.1f} Hz')
    axs[4].set_title('5. Transformation: Power Spectral Density (Frequency Domain)', fontsize=12, fontweight='bold')
    axs[4].set_ylabel('Power/Freq (dB/Hz)')
    axs[4].set_xlabel('Frequency (Hz)')
    axs[4].set_xlim(0, 50) # Limit x-axis to show relevant ECG frequencies
    axs[4].legend(loc='upper right')
    axs[4].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    filename = 'traditional_ecg_pipeline_with_features.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nFigure successfully saved as '{filename}' at 300 dpi.")
    
    plt.show()

if __name__ == "__main__":
    traditional_ecg_pipeline_demo()