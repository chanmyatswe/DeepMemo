"""
Doppler Simulator using Classical Clarke Model
Based on Section 3.1.3, 3.6, and Equations 3.11, 3.12
"""

import numpy as np
from tqdm import tqdm

class DopplerSimulator:
    """
    Implements Classical Clarke model for temporal CSI generation
    
    Key features:
    - Isotropic scattering (uniform AoA distribution)
    - Multipath Rayleigh fading per path
    - Doppler shift: fD,p = (v/c) * fc * cos(θp)
    - Temporal evolution: hp(t) = αp * exp(j*2π*fD,p*t)
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self.fc = config.CARRIER_FREQUENCY
        self.c = config.SPEED_OF_LIGHT
        self.num_paths = config.NUM_PATHS
        
        # For reproducibility
        np.random.seed(config.RANDOM_SEED)
    
    def compute_doppler_shift(self, velocity_ms, theta):
        """
        Compute Doppler shift for a single path
        
        Equation 3.11: fD,p = (v/c) * fc * cos(θp)
        
        Args:
            velocity_ms: User velocity in m/s
            theta: Angle of arrival in radians
            
        Returns:
            Doppler frequency shift in Hz
        """
        return (velocity_ms / self.c) * self.fc * np.cos(theta)
    
    def compute_max_doppler(self, velocity_ms):
        """
        Compute maximum Doppler frequency
        
        Args:
            velocity_ms: User velocity in m/s
            
        Returns:
            Maximum Doppler in Hz (when cos(θ) = 1)
        """
        return (velocity_ms / self.c) * self.fc
    
    def compute_coherence_time(self, velocity_ms):
        """
        Compute channel coherence time
        
        Equation 3.11: Tc ≈ 0.423 / fD,max
        
        Args:
            velocity_ms: User velocity in m/s
            
        Returns:
            Coherence time in seconds
        """
        fd_max = self.compute_max_doppler(velocity_ms)
        if fd_max == 0:
            return float('inf')  # Static channel
        return 0.423 / fd_max
    
    def generate_path_parameters(self, num_paths=None):
        """
        Generate random parameters for multipath components
        
        Returns:
            Dictionary with:
            - aoa: Angles of arrival (uniform [0, 2π])
            - alpha: Complex Rayleigh gains
            - delays: Path delays (for frequency-selective fading)
        """
        if num_paths is None:
            num_paths = self.num_paths
        
        # Angles of arrival: uniform [0, 2π] for isotropic scattering
        aoa = np.random.uniform(0, 2*np.pi, num_paths)
        
        # Complex Rayleigh gains: CN(0, 1)
        # Real and imaginary parts are independent N(0, 1/2)
        alpha_real = np.random.normal(0, 1/np.sqrt(2), num_paths)
        alpha_imag = np.random.normal(0, 1/np.sqrt(2), num_paths)
        alpha = alpha_real + 1j * alpha_imag
        
        # Normalize total power to 1
        alpha = alpha / np.sqrt(np.sum(np.abs(alpha)**2))
        
        # Path delays (for frequency-selective effects)
        delays = np.sort(np.random.exponential(1.0, num_paths))
        delays = delays / np.max(delays)  # Normalize to [0, 1]
        
        return {
            'aoa': aoa,
            'alpha': alpha,
            'delays': delays
        }
    
    def generate_temporal_csi_single_user(self, H_static, velocity_ms, 
                                         num_time_steps, delta_t, 
                                         path_params=None):
        """
        Generate temporal CSI sequence for a single user with Doppler effects
        
        Equation 3.12:
        hp(t) = αp * exp(j*2π*fD,p*t)
        h(t) = Σ hp(t)
        
        Args:
            H_static: Static channel from DeepMIMO (Nr, Nt, Nk)
            velocity_ms: User velocity in m/s
            num_time_steps: Number of temporal snapshots (T_total)
            delta_t: Time step interval in seconds (Δt)
            path_params: Pre-generated path parameters (optional)
            
        Returns:
            H_temporal: Time-varying channel (T_total, Nr, Nt, Nk)
        """
        # Get dimensions
        if H_static.ndim == 3:
            Nr, Nt, Nk = H_static.shape
        elif H_static.ndim == 1:  # SISO case
            Nk = H_static.shape[0]
            Nr, Nt = 1, 1
            H_static = H_static.reshape(1, 1, -1)
        else:
            raise ValueError(f"Unexpected H_static shape: {H_static.shape}")
        
        # Generate path parameters if not provided
        if path_params is None:
            path_params = self.generate_path_parameters()
        
        aoa = path_params['aoa']
        alpha = path_params['alpha']
        
        # Compute Doppler shifts for each path
        doppler_shifts = np.array([
            self.compute_doppler_shift(velocity_ms, theta) 
            for theta in aoa
        ])
        
        # Initialize temporal channel
        H_temporal = np.zeros((num_time_steps, Nr, Nt, Nk), dtype=complex)
        
        # Generate temporal evolution
        for t in range(num_time_steps):
            time = t * delta_t
            
            # Compute phase shift for each path: exp(j*2π*fD,p*t)
            phase_shifts = np.exp(1j * 2 * np.pi * doppler_shifts * time)
            
            # Apply path gains and phases: hp(t) = αp * exp(j*2π*fD,p*t)
            h_paths = alpha * phase_shifts
            
            # Sum over all paths to get total channel at time t
            # This is the Clarke model: h(t) = Σ_p hp(t)
            h_total = np.sum(h_paths)
            
            # Apply to static channel (broadcast across antennas/subcarriers)
            H_temporal[t] = H_static * h_total
        
        return H_temporal
    
    def generate_temporal_dataset(self, deepmimo_dataset, velocity_class, 
                                  num_users=None, show_progress=True):
        """
        Generate temporal dataset for multiple users at a given velocity
        
        Args:
            deepmimo_dataset: DeepMIMO dataset object
            velocity_class: Velocity class name (e.g., 'urban_60')
            num_users: Number of users to process (None = use config)
            show_progress: Show progress bar
            
        Returns:
            Dictionary with:
            - H_temporal: (num_users, T_total, Nr, Nt, Nk)
            - velocity_ms: Velocity in m/s
            - velocity_kmh: Velocity in km/h
            - fd_max: Maximum Doppler frequency
            - Tc: Coherence time
        """
        if num_users is None:
            num_users = self.config.NUM_USERS_PER_VELOCITY
        
        # Get velocity
        velocity_kmh = self.config.VELOCITY_CLASSES[velocity_class]
        velocity_ms = self.config.kmh_to_ms(velocity_kmh)
        
        # Get temporal parameters
        T_total = self.config.TOTAL_TIME_STEPS
        delta_t = self.config.SAMPLING_INTERVAL
        
        # Get channel data
        channels = deepmimo_dataset[0]['user']['channel']
        num_available_users = len(channels)
        
        if num_users > num_available_users:
            print(f"Warning: Requested {num_users} users but only "
                  f"{num_available_users} available. Using {num_available_users}.")
            num_users = num_available_users
        
        # Determine output shape
        H_sample = channels[0]
        if H_sample.ndim == 1:  # SISO
            Nr, Nt, Nk = 1, 1, len(H_sample)
        else:
            Nr, Nt, Nk = H_sample.shape
        
        # Initialize output
        H_temporal_all = np.zeros((num_users, T_total, Nr, Nt, Nk), 
                                  dtype=complex)
        
        # Generate for each user
        iterator = range(num_users)
        if show_progress:
            iterator = tqdm(iterator, 
                           desc=f"Generating {velocity_class} ({velocity_kmh:.0f} km/h)")
        
        for user_idx in iterator:
            H_static = channels[user_idx]
            
            # Generate temporal CSI with Doppler
            H_temporal = self.generate_temporal_csi_single_user(
                H_static, velocity_ms, T_total, delta_t
            )
            
            H_temporal_all[user_idx] = H_temporal
        
        # Compute Doppler statistics
        fd_max = self.compute_max_doppler(velocity_ms)
        Tc = self.compute_coherence_time(velocity_ms)
        
        return {
            'H_temporal': H_temporal_all,
            'velocity_class': velocity_class,
            'velocity_ms': velocity_ms,
            'velocity_kmh': velocity_kmh,
            'fd_max': fd_max,
            'Tc': Tc,
            'num_users': num_users,
            'T_total': T_total,
            'delta_t': delta_t
        }
    
    def create_sliding_windows(self, H_temporal, window_size=None, 
                              pred_horizon=None, stride=None):
        """
        Create sliding window sequences for supervised learning
        
        Equation 3.13:
        {H[t], ..., H[t+T-1]} → H[t+T]  (sequence-to-one)
        
        Args:
            H_temporal: Temporal CSI (num_users, T_total, Nr, Nt, Nk)
            window_size: Input sequence length (T)
            pred_horizon: Number of future frames to predict (Q)
            stride: Sliding window stride
            
        Returns:
            X_sequences: Input sequences (N, T, Nr, Nt, Nk)
            Y_sequences: Target sequences (N, Q, Nr, Nt, Nk)
        """
        if window_size is None:
            window_size = self.config.SEQUENCE_LENGTH
        if pred_horizon is None:
            pred_horizon = self.config.PREDICTION_HORIZON
        if stride is None:
            stride = self.config.STRIDE
        
        num_users, T_total, Nr, Nt, Nk = H_temporal.shape
        
        X_sequences = []
        Y_sequences = []
        
        for user_idx in range(num_users):
            H_user = H_temporal[user_idx]  # (T_total, Nr, Nt, Nk)
            
            # Slide window across time
            for start in range(0, T_total - window_size - pred_horizon + 1, stride):
                # Input: past window_size frames
                X_seq = H_user[start:start + window_size]
                
                # Target: next pred_horizon frame(s)
                Y_seq = H_user[start + window_size:start + window_size + pred_horizon]
                
                X_sequences.append(X_seq)
                Y_sequences.append(Y_seq)
        
        X_sequences = np.array(X_sequences)
        Y_sequences = np.array(Y_sequences)
        
        return X_sequences, Y_sequences
    
    def print_doppler_info(self, velocity_ms):
        """Print Doppler-related information for a given velocity"""
        velocity_kmh = velocity_ms * 3.6
        fd_max = self.compute_max_doppler(velocity_ms)
        Tc = self.compute_coherence_time(velocity_ms)
        
        print(f"\nDoppler Information for v = {velocity_kmh:.1f} km/h:")
        print(f"  Velocity: {velocity_ms:.2f} m/s")
        print(f"  Max Doppler (fD,max): {fd_max:.2f} Hz")
        print(f"  Coherence Time (Tc): {Tc*1e3:.2f} ms")
        print(f"  Sampling Interval (Δt): {self.config.SAMPLING_INTERVAL*1e6:.2f} μs")
        print(f"  Samples within Tc: {int(Tc / self.config.SAMPLING_INTERVAL)}")


def test_doppler_simulator():
    """Test Doppler simulator"""
    from config import config
    import matplotlib.pyplot as plt
    
    print("\n" + "="*60)
    print("TESTING DOPPLER SIMULATOR")
    print("="*60)
    
    simulator = DopplerSimulator(config)
    
    # Test with synthetic static channel (SISO)
    H_static = np.ones(512, dtype=complex)  # 512 subcarriers
    
    # Test different velocities
    velocities = [1/3.6, 30/3.6, 100/3.6, 250/3.6]  # m/s
    colors = ['blue', 'green', 'orange', 'red']
    labels = ['1 km/h', '30 km/h', '100 km/h', '250 km/h']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Doppler Effects at Different Velocities (Clarke Model)', 
                 fontsize=14, fontweight='bold')
    
    for idx, (vel_ms, color, label) in enumerate(zip(velocities, colors, labels)):
        # Print Doppler info
        simulator.print_doppler_info(vel_ms)
        
        # Generate temporal CSI
        H_temporal = simulator.generate_temporal_csi_single_user(
            H_static, vel_ms, num_time_steps=200, 
            delta_t=config.SAMPLING_INTERVAL
        )
        
        # Plot magnitude and phase evolution
        time_axis = np.arange(200) * config.SAMPLING_INTERVAL * 1e3  # ms
        
        # Magnitude plot
        ax1 = axes[0, 0] if idx < 2 else axes[0, 1]
        if idx % 2 == 0:
            magnitude = np.abs(H_temporal[:, 0, 0, 0])
            ax1.plot(time_axis, magnitude, color=color, label=label, linewidth=2)
            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('|H(t)|')
            ax1.set_title('Channel Magnitude Evolution')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # Phase plot
        ax2 = axes[1, 0] if idx < 2 else axes[1, 1]
        if idx % 2 == 0:
            phase = np.angle(H_temporal[:, 0, 0, 0])
            ax2.plot(time_axis, phase, color=color, label=label, linewidth=2)
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('∠H(t) (radians)')
            ax2.set_title('Channel Phase Evolution (Doppler Shift)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
    
    plt.tight_layout()
    save_path = config.VIZ_DIR + '/doppler_effects.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {save_path}")
    plt.close()
    
    print("="*60 + "\n")


if __name__ == "__main__":
    test_doppler_simulator()