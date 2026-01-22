"""
Channel Models: AWGN, Rayleigh, and Rician Fading
Based on Section 3.1.4 and Equation 3.1
"""

import numpy as np

class ChannelModels:
    """
    Implements wireless channel models with toggleable components
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration object with channel settings
        """
        self.config = config
        self.enable_awgn = config.CHANNEL_CONFIG['enable_awgn']
        self.enable_rayleigh = config.CHANNEL_CONFIG['enable_rayleigh']
        self.enable_rician = config.CHANNEL_CONFIG['enable_rician']
        self.rician_k_factor = config.CHANNEL_CONFIG['rician_k_factor']
    
    def add_awgn(self, signal, snr_db):
        """
        Add Additive White Gaussian Noise
        
        Args:
            signal: Complex-valued signal (any shape)
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Noisy signal with same shape
        """
        if not self.enable_awgn:
            return signal
        
        # Calculate signal power
        signal_power = np.mean(np.abs(signal)**2)
        
        # Convert SNR from dB to linear scale
        snr_linear = 10**(snr_db / 10)
        
        # Calculate noise power
        noise_power = signal_power / snr_linear
        
        # Generate complex Gaussian noise
        noise_std = np.sqrt(noise_power / 2)  # Divide by 2 for complex
        noise_real = np.random.normal(0, noise_std, signal.shape)
        noise_imag = np.random.normal(0, noise_std, signal.shape)
        noise = noise_real + 1j * noise_imag
        
        return signal + noise
    
    def generate_rayleigh_fading(self, shape):
        """
        Generate Rayleigh fading coefficients (NLOS scenario)
        
        Rayleigh assumes rich scattering with no dominant path.
        Each path has independent complex Gaussian amplitude.
        
        Args:
            shape: Desired output shape (e.g., (Nr, Nt, Nk))
            
        Returns:
            Complex Rayleigh fading coefficients
        """
        if not self.enable_rayleigh:
            return np.ones(shape, dtype=complex)
        
        # Generate independent complex Gaussian variables
        # Each component: N(0, 1/2) for normalized power
        real_part = np.random.normal(0, 1/np.sqrt(2), shape)
        imag_part = np.random.normal(0, 1/np.sqrt(2), shape)
        
        fading_coeff = real_part + 1j * imag_part
        
        return fading_coeff
    
    def generate_rician_fading(self, shape, k_factor_db=None):
        """
        Generate Rician fading coefficients (LOS + scattered)
        
        Rician model includes both line-of-sight (LOS) and scattered components.
        K-factor determines the ratio of LOS to scattered power.
        
        Args:
            shape: Desired output shape
            k_factor_db: Rician K-factor in dB (None = use config default)
            
        Returns:
            Complex Rician fading coefficients
        """
        if not self.enable_rician:
            return np.ones(shape, dtype=complex)
        
        if k_factor_db is None:
            k_factor_db = self.rician_k_factor
        
        # Convert K from dB to linear
        K = 10**(k_factor_db / 10)
        
        # LOS component (deterministic)
        # Normalized so that total power = 1
        los_amplitude = np.sqrt(K / (K + 1))
        los_component = los_amplitude * np.ones(shape, dtype=complex)
        
        # Scattered component (Rayleigh)
        scatter_amplitude = np.sqrt(1 / (K + 1))
        real_part = np.random.normal(0, 1/np.sqrt(2), shape)
        imag_part = np.random.normal(0, 1/np.sqrt(2), shape)
        scatter_component = scatter_amplitude * (real_part + 1j * imag_part)
        
        # Total Rician fading
        rician_coeff = los_component + scatter_component
        
        return rician_coeff
    
    def apply_fading(self, channel, fading_type='rayleigh'):
        """
        Apply fading to channel
        
        Args:
            channel: Base channel coefficients
            fading_type: 'rayleigh' or 'rician'
            
        Returns:
            Channel with fading applied
        """
        if fading_type == 'rayleigh' and self.enable_rayleigh:
            fading = self.generate_rayleigh_fading(channel.shape)
            return channel * fading
        elif fading_type == 'rician' and self.enable_rician:
            fading = self.generate_rician_fading(channel.shape)
            return channel * fading
        else:
            return channel
    
    def apply_channel_effects(self, channel, snr_db, fading_type='rayleigh'):
        """
        Apply complete channel model: Fading + AWGN
        
        This is the main method to use for processing channels
        
        Args:
            channel: Input channel (T x Nr x Nt x Nk)
            snr_db: SNR in dB
            fading_type: 'rayleigh' or 'rician'
            
        Returns:
            Channel with fading and noise applied
        """
        # Step 1: Apply fading (if enabled)
        channel_faded = self.apply_fading(channel, fading_type)
        
        # Step 2: Add AWGN (if enabled)
        channel_noisy = self.add_awgn(channel_faded, snr_db)
        
        return channel_noisy
    
    def get_channel_type_string(self):
        """Return string describing active channel model"""
        components = []
        if self.enable_rayleigh:
            components.append("Rayleigh")
        if self.enable_rician:
            components.append(f"Rician(K={self.rician_k_factor}dB)")
        if self.enable_awgn:
            components.append("AWGN")
        
        if not components:
            return "Ideal"
        return " + ".join(components)


class FrequencySelectiveChannel:
    """
    Models frequency-selective multipath channel
    Based on Equation 3.1: h(t) = Σ αp * exp(j2πfD,p*t)
    """
    
    def __init__(self, num_paths, num_subcarriers):
        """
        Args:
            num_paths: Number of multipath components (P)
            num_subcarriers: Number of OFDM subcarriers (Nk)
        """
        self.num_paths = num_paths
        self.num_subcarriers = num_subcarriers
        
        # Generate random path delays (normalized to [0, 1])
        self.path_delays = np.sort(np.random.rand(num_paths))
        
        # Generate random path gains (Rayleigh distributed)
        self.path_gains = self._generate_path_gains()
    
    def _generate_path_gains(self):
        """Generate complex path gains with exponential power delay profile"""
        # Exponential power decay
        powers = np.exp(-2 * np.arange(self.num_paths))
        powers = powers / np.sum(powers)  # Normalize
        
        # Complex Gaussian gains with appropriate power
        real_part = np.random.normal(0, 1, self.num_paths)
        imag_part = np.random.normal(0, 1, self.num_paths)
        gains = (real_part + 1j * imag_part) * np.sqrt(powers / 2)
        
        return gains
    
    def get_frequency_response(self):
        """
        Compute frequency-domain channel response across subcarriers
        
        Returns:
            H[k]: Channel frequency response (Nk,) complex array
        """
        H = np.zeros(self.num_subcarriers, dtype=complex)
        
        # Sum contribution of each path
        for p in range(self.num_paths):
            # Frequency response: H[k] = α_p * exp(-j2π*k*τ_p)
            k = np.arange(self.num_subcarriers)
            H += self.path_gains[p] * np.exp(-1j * 2 * np.pi * k * 
                                              self.path_delays[p] / 
                                              self.num_subcarriers)
        
        return H


def test_channel_models():
    """Test channel models with visualization"""
    from config import config
    import matplotlib.pyplot as plt
    
    print("\n" + "="*60)
    print("TESTING CHANNEL MODELS")
    print("="*60)
    
    # Create channel model
    channel_model = ChannelModels(config)
    print(f"Active channel model: {channel_model.get_channel_type_string()}")
    
    # Test signal
    signal_shape = (100, 4, 4, 64)  # (T, Nr, Nt, Nk)
    signal = np.ones(signal_shape, dtype=complex)
    
    # Test different SNR levels
    snr_levels = [5, 10, 15, 20]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Channel Model Effects at Different SNR Levels', fontsize=14)
    
    for idx, snr_db in enumerate(snr_levels):
        ax = axes[idx // 2, idx % 2]
        
        # Apply channel effects
        noisy_signal = channel_model.apply_channel_effects(signal, snr_db)
        
        # Plot magnitude of first subcarrier over time
        magnitude = np.abs(noisy_signal[:, 0, 0, 0])
        ax.plot(magnitude, label=f'SNR = {snr_db} dB')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('|H| Magnitude')
        ax.set_title(f'SNR = {snr_db} dB')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    save_path = config.VIZ_DIR + '/channel_model_test.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {save_path}")
    plt.close()
    
    # Test frequency-selective channel
    print("\n" + "-"*60)
    print("Testing Frequency-Selective Channel")
    fs_channel = FrequencySelectiveChannel(num_paths=20, num_subcarriers=512)
    H_freq = fs_channel.get_frequency_response()
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(np.abs(H_freq))
    plt.xlabel('Subcarrier Index')
    plt.ylabel('|H[k]|')
    plt.title('Channel Frequency Response (Magnitude)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.angle(H_freq))
    plt.xlabel('Subcarrier Index')
    plt.ylabel('∠H[k] (radians)')
    plt.title('Channel Frequency Response (Phase)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = config.VIZ_DIR + '/frequency_selective_channel.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved frequency response to: {save_path}")
    plt.close()
    
    print("="*60 + "\n")


if __name__ == "__main__":
    test_channel_models()