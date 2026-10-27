import numpy as np
import matplotlib.pyplot as plt

def demonstrate_nyquist_aliasing():
    # Signal and sampling parameters
    f_signal = 40.0       # True signal frequency (Hz)
    fs_adequate = 100.0   # Adequate sampling frequency (Hz)
    fs_inadequate = 50.0  # Inadequate sampling frequency (Hz) -> causes aliasing
    duration = 1.0        # Duration in seconds
    
    # Time vectors
    t_continuous = np.linspace(0, duration, 1000)
    t_adequate = np.arange(0, duration, 1/fs_adequate)
    t_inadequate = np.arange(0, duration, 1/fs_inadequate)
    
    # Signal generation
    signal_continuous = np.sin(2 * np.pi * f_signal * t_continuous)
    signal_adequate = np.sin(2 * np.pi * f_signal * t_adequate)
    signal_inadequate = np.sin(2 * np.pi * f_signal * t_inadequate)
    
    # Calculate aliased frequency: f_alias = |f_signal - N * fs|
    # For fs = 50 and f_signal = 40, the first alias is |40 - 50| = 10 Hz
    f_aliased = abs(f_signal - fs_inadequate)
    
    # Console output 
    print("="*45)
    print("NYQUIST-SHANNON ALIASING DEMONSTRATION")
    print("="*45)
    print(f"True Signal Frequency:      {f_signal} Hz")
    print(f"Inadequate Sampling Rate:   {fs_inadequate} Hz")
    print(f"Aliased Frequency Observed: {f_aliased} Hz")
    print("="*45)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # --- TOP PLOT: ADEQUATE SAMPLING ---
    ax1.plot(t_continuous, signal_continuous, color='gray', linewidth=2, label='True Continuous 40 Hz')
    ax1.stem(t_adequate, signal_adequate, linefmt='b-', markerfmt='bo', basefmt=' ', 
             label=f'Sampled at {fs_adequate} Hz (Adequate, $f_s > 2f_{{max}}$)')
             
    ax1.set_title('Adequate Sampling: True Signal Preserved', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_ylim(-1.5, 1.5)
    
    # --- BOTTOM PLOT: INADEQUATE SAMPLING (ALIASING) ---
    # 'gray'
    ax2.plot(t_continuous, signal_continuous, color='gray', linewidth=2, label='True Continuous 40 Hz')
    
    # The theoretical aliased wave (red dashed)
    ax2.plot(t_continuous, np.sin(2 * np.pi * f_aliased * t_continuous), color='red', linestyle='--', linewidth=2,
             label=f'Theoretical Aliased {f_aliased} Hz Wave')
    
    # The raw sampled points (red stems, no markers to avoid clutter)
    ax2.stem(t_inadequate, signal_inadequate, linefmt='r-', markerfmt=' ', basefmt=' ',
             label=f'Sampled Points at {fs_inadequate} Hz')
    
    # The reconstructed path connecting the samples (high contrast markers)
    ax2.plot(t_inadequate, signal_inadequate, color='darkblue', linestyle='-', marker='o', markersize=8,
             markerfacecolor='white', markeredgecolor='darkblue', markeredgewidth=2,
             label='Reconstructed Path (The Illusion)')
    
    ax2.set_title('Inadequate Sampling: Aliasing Artifact Introduced', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Amplitude', fontsize=12)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    
    # Save the figure as a high-resolution PNG
    filename = 'nyquist_aliasing_demonstration.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nFigure successfully saved as '{filename}' at 300 dpi.")
    
    plt.show()

if __name__ == "__main__":
    demonstrate_nyquist_aliasing()