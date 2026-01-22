"""
Visualization utilities for temporal dataset analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import os


class DatasetVisualizer:
    """Generate plots and visualizations for temporal dataset"""
    
    def __init__(self, config):
        self.config = config
        
    def plot_temporal_evolution(self, temporal_data, save_prefix="temporal"):
        """
        Plot how channel magnitude and phase evolve over time for different velocities
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Channel Temporal Evolution Across Velocities', 
                     fontsize=16, fontweight='bold')
        
        for idx, (vel_class, data) in enumerate(temporal_data.items()):
            if idx >= 6:  # Max 6 subplots
                break
            
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # Get one sequence
            X_seq = data['X_sequences'][0]  # (T, Nr, Nt, Nk)
            
            # Extract first antenna/subcarrier
            if X_seq.shape[1] == 1:  # SISO
                channel_seq = X_seq[:, 0, 0, 0]
            else:  # MIMO
                channel_seq = X_seq[:, 0, 0, 0]
            
            # Time axis
            time_ms = np.arange(len(channel_seq)) * self.config.SAMPLING_INTERVAL * 1e3
            
            # Plot magnitude and phase
            ax_phase = ax.twinx()
            
            mag = np.abs(channel_seq)
            phase = np.angle(channel_seq)
            
            l1 = ax.plot(time_ms, mag, 'b-', linewidth=2, label='Magnitude')
            ax.set_ylabel('|H(t)|', color='b', fontsize=11)
            ax.tick_params(axis='y', labelcolor='b')
            
            l2 = ax_phase.plot(time_ms, phase, 'r-', linewidth=1.5, 
                              alpha=0.7, label='Phase')
            ax_phase.set_ylabel('∠H(t) (rad)', color='r', fontsize=11)
            ax_phase.tick_params(axis='y', labelcolor='r')
            
            ax.set_xlabel('Time (ms)', fontsize=11)
            ax.set_title(f"{vel_class}\n{data['velocity_kmh']:.0f} km/h | "
                        f"fD={data['fd_max']:.1f} Hz", fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Combined legend
            lns = l1 + l2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        save_path = os.path.join(self.config.VIZ_DIR, f'{save_prefix}_evolution.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")
        
    def plot_doppler_spectrum(self, temporal_data, save_prefix="doppler"):
        """
        Plot Doppler power spectral density for different velocities
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Doppler Power Spectral Density', 
                     fontsize=16, fontweight='bold')
        
        for idx, (vel_class, data) in enumerate(temporal_data.items()):
            if idx >= 6:
                break
            
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # Get one long sequence
            X_seq = data['X_sequences'][0]
            
            if X_seq.shape[1] == 1:  # SISO
                channel_seq = X_seq[:, 0, 0, 0]
            else:
                channel_seq = X_seq[:, 0, 0, 0]
            
            # Compute FFT
            fft_vals = np.fft.fft(channel_seq)
            fft_freq = np.fft.fftfreq(len(channel_seq), 
                                      d=self.config.SAMPLING_INTERVAL)
            
            # Power spectrum
            power = np.abs(fft_vals)**2
            
            # Shift to center zero frequency
            fft_freq_shifted = np.fft.fftshift(fft_freq)
            power_shifted = np.fft.fftshift(power)
            
            # Plot
            ax.plot(fft_freq_shifted, 10*np.log10(power_shifted + 1e-12), 
                   linewidth=2)
            ax.set_xlabel('Frequency (Hz)', fontsize=11)
            ax.set_ylabel('Power (dB)', fontsize=11)
            ax.set_title(f"{vel_class} ({data['velocity_kmh']:.0f} km/h)\n"
                        f"fD,max = {data['fd_max']:.1f} Hz", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axvline(x=data['fd_max'], color='r', linestyle='--', 
                      linewidth=2, alpha=0.7, label=f"fD,max")
            ax.axvline(x=-data['fd_max'], color='r', linestyle='--', 
                      linewidth=2, alpha=0.7)
            ax.legend(fontsize=9)
            
            # Zoom to Doppler region
            ax.set_xlim(-3*data['fd_max'], 3*data['fd_max'])
        
        plt.tight_layout()
        save_path = os.path.join(self.config.VIZ_DIR, f'{save_prefix}_spectrum.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    
    def plot_channel_correlation(self, temporal_data, save_prefix="correlation"):
        """
        Plot temporal autocorrelation for different velocities
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Temporal Autocorrelation vs Coherence Time', 
                     fontsize=16, fontweight='bold')
        
        for idx, (vel_class, data) in enumerate(temporal_data.items()):
            if idx >= 6:
                break
            
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # Get sequence
            X_seq = data['X_sequences'][0]
            
            if X_seq.shape[1] == 1:
                channel_seq = X_seq[:, 0, 0, 0]
            else:
                channel_seq = X_seq[:, 0, 0, 0]
            
            # Compute autocorrelation
            autocorr = np.correlate(channel_seq, channel_seq, mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Time lags
            lags_ms = np.arange(len(autocorr)) * self.config.SAMPLING_INTERVAL * 1e3
            
            # Plot
            ax.plot(lags_ms, np.abs(autocorr), linewidth=2)
            ax.set_xlabel('Time Lag (ms)', fontsize=11)
            ax.set_ylabel('|Autocorrelation|', fontsize=11)
            ax.set_title(f"{vel_class} ({data['velocity_kmh']:.0f} km/h)\n"
                        f"Tc = {data['Tc']*1e3:.2f} ms", fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Mark coherence time
            Tc_ms = data['Tc'] * 1e3
            ax.axvline(x=Tc_ms, color='r', linestyle='--', 
                      linewidth=2, alpha=0.7, label=f"Tc = {Tc_ms:.2f} ms")
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
            ax.legend(fontsize=9)
            
            # Reasonable x-axis limit
            ax.set_xlim(0, min(5*Tc_ms, lags_ms[-1]))
        
        plt.tight_layout()
        save_path = os.path.join(self.config.VIZ_DIR, 
                                f'{save_prefix}_autocorr.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    
    def plot_dataset_summary(self, temporal_data, save_prefix="summary"):
        """
        Summary plot: velocity vs Doppler vs coherence time
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        velocities = []
        fd_maxs = []
        tcs = []
        labels = []
        
        for vel_class, data in temporal_data.items():
            velocities.append(data['velocity_kmh'])
            fd_maxs.append(data['fd_max'])
            tcs.append(data['Tc'] * 1e3)  # Convert to ms
            labels.append(vel_class)
        
        # Sort by velocity
        sort_idx = np.argsort(velocities)
        velocities = np.array(velocities)[sort_idx]
        fd_maxs = np.array(fd_maxs)[sort_idx]
        tcs = np.array(tcs)[sort_idx]
        labels = np.array(labels)[sort_idx]
        
        # Plot 1: Velocity vs Max Doppler
        ax1 = axes[0]
        ax1.plot(velocities, fd_maxs, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Velocity (km/h)', fontsize=12)
        ax1.set_ylabel('Max Doppler Frequency (Hz)', fontsize=12)
        ax1.set_title('Velocity vs Maximum Doppler Shift', fontsize=13, 
                     fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        for i, label in enumerate(labels):
            ax1.annotate(label, (velocities[i], fd_maxs[i]), 
                        textcoords="offset points", xytext=(0,10), 
                        ha='center', fontsize=8)
        
        # Plot 2: Velocity vs Coherence Time
        ax2 = axes[1]
        ax2.semilogy(velocities, tcs, 's-', linewidth=2, markersize=8, color='red')
        ax2.set_xlabel('Velocity (km/h)', fontsize=12)
        ax2.set_ylabel('Coherence Time (ms)', fontsize=12)
        ax2.set_title('Velocity vs Channel Coherence Time', fontsize=13, 
                     fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')
        
        for i, label in enumerate(labels):
            ax2.annotate(label, (velocities[i], tcs[i]), 
                        textcoords="offset points", xytext=(0,10), 
                        ha='center', fontsize=8)
        
        plt.tight_layout()
        save_path = os.path.join(self.config.VIZ_DIR, f'{save_prefix}_overview.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    
    def visualize_dataset(self, temporal_data):
        """
        Generate all visualization plots
        
        Args:
            temporal_data: Dictionary from TemporalDatasetGenerator
        """
        print("\nGenerating visualizations...")
        
        self.plot_temporal_evolution(temporal_data)
        self.plot_doppler_spectrum(temporal_data)
        self.plot_channel_correlation(temporal_data)
        self.plot_dataset_summary(temporal_data)
        
        print(f"\n✓ All visualizations saved to: {self.config.VIZ_DIR}")