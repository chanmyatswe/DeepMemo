"""
Configuration file for Temporal DeepMIMO Dataset Generation
Based on methodology Chapter 3 and Table 3.2
"""

import numpy as np

class Config:
    """Central configuration for dataset generation"""
    
    # ==================== DeepMIMO Scenario ====================
    SCENARIO = 'O1_3p5'  # 3.5 GHz scenario
    ACTIVE_BS = [1]      # Base station index
    
    # User grid selection
    USER_ROW_FIRST = 1
    USER_ROW_LAST = 100
    USER_ROW_SUBSAMPLING = 1
    
    # ==================== MIMO Configuration ====================
    # Start with SISO, then scale to MIMO
    ANTENNA_CONFIG = 'SISO'  # Options: 'SISO', '2x2', '4x4', '8x8'
    
    # Antenna mapping
    ANTENNA_MAP = {
        'SISO': {'Nt': 1, 'Nr': 1},
        '2x2': {'Nt': 2, 'Nr': 2},
        '4x4': {'Nt': 4, 'Nr': 4},
        '8x8': {'Nt': 8, 'Nr': 8}
    }
    
    @property
    def Nt(self):
        return self.ANTENNA_MAP[self.ANTENNA_CONFIG]['Nt']
    
    @property
    def Nr(self):
        return self.ANTENNA_MAP[self.ANTENNA_CONFIG]['Nr']
    
    # ==================== OFDM Parameters (Table 3.2) ====================
    CARRIER_FREQUENCY = 3.5e9  # 3.5 GHz (fc)
    SUBCARRIER_SPACING = 60e3  # 60 kHz (Δf) - 5G NR numerology
    NUM_SUBCARRIERS = 512      # Number of OFDM subcarriers (Nk)
    BANDWIDTH = 50e6           # 50 MHz
    
    # OFDM symbol duration (Ts ≈ 1/Δf)
    OFDM_SYMBOL_DURATION = 1 / SUBCARRIER_SPACING  # ~16.7 μs
    
    # ==================== Temporal Parameters ====================
    # Sampling interval (Δt = m × Ts, where m ∈ {1,2,4})
    SAMPLING_MULTIPLIER = 1  # Start with m=1 (finest resolution)
    SAMPLING_INTERVAL = SAMPLING_MULTIPLIER * OFDM_SYMBOL_DURATION  # Δt
    
    # Sequence parameters
    TOTAL_TIME_STEPS = 100      # Total snapshots per user (T_total)
    SEQUENCE_LENGTH = 10        # Window size (T) - Table 3.2
    PREDICTION_HORIZON = 1      # Predict next Q frames
    STRIDE = 1                  # Sliding window stride
    
    # ==================== Velocity Classes (Table 3.1) ====================
    # Velocities in km/h, converted to m/s
    VELOCITY_CLASSES = {
        'pedestrian_1': 1.0,       # 1 km/h ≈ 0.28 m/s
        'pedestrian_5': 5.0,       # 5 km/h ≈ 1.39 m/s
        'urban_30': 30.0,          # 30 km/h ≈ 8.33 m/s
        'urban_60': 60.0,          # 60 km/h ≈ 16.67 m/s
        'urban_100': 100.0,        # 100 km/h ≈ 27.78 m/s
        'high_speed_250': 250.0,   # 250 km/h ≈ 69.44 m/s
        # 'high_speed_350': 350.0, # Optional: 350 km/h ≈ 97.22 m/s
    }
    
    @staticmethod
    def kmh_to_ms(velocity_kmh):
        """Convert km/h to m/s"""
        return velocity_kmh / 3.6
    
    @property
    def velocities_ms(self):
        """Get velocities in m/s"""
        return {k: self.kmh_to_ms(v) for k, v in self.VELOCITY_CLASSES.items()}
    
    # ==================== Clarke Model Parameters ====================
    SPEED_OF_LIGHT = 3e8       # c (m/s)
    NUM_PATHS = 20             # Number of multipath components (P)
    
    # ==================== Channel Models ====================
    CHANNEL_CONFIG = {
        'enable_awgn': True,       # Additive White Gaussian Noise
        'enable_rayleigh': True,   # Multipath fading (NLOS)
        'enable_rician': False,    # LOS + scattered (set True for Rician)
        'rician_k_factor': 3,      # K-factor in dB (when Rician enabled)
    }
    
    # ==================== SNR Configuration ====================
    # SNR levels to evaluate (dB) - Table 3.2 mentions 5-20 dB range
    SNR_DB_RANGE = [5, 10, 15, 20]  # Can be single value or list
    DEFAULT_SNR_DB = 10              # Default SNR for generation
    
    # SNR-Velocity coupling (optional)
    SNR_VELOCITY_COUPLING = False  # Set True for realistic degradation
    
    # If coupling enabled, SNR decreases with velocity
    # High velocity → more Doppler → harder tracking → lower effective SNR
    SNR_VELOCITY_MAPPING = {
        'pedestrian_1': 20,     # Best SNR at low speed
        'pedestrian_5': 18,
        'urban_30': 15,
        'urban_60': 12,
        'urban_100': 10,
        'high_speed_250': 8,    # Worst SNR at high speed
    }
    
    # ==================== Dataset Generation ====================
    NUM_USERS_PER_VELOCITY = 20  # Start small: 20 users/velocity for testing
                                  # Increase to 100-200 for production
    
    # Train/Val/Test split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # ==================== Paths ====================
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    VIZ_DIR = os.path.join(DATA_DIR, 'visualizations')
    
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    
    # Create directories if they don't exist
    for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VIZ_DIR, LOG_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # ==================== Utility Methods ====================
    def get_doppler_frequency(self, velocity_ms):
        """
        Calculate maximum Doppler frequency (Equation 3.11)
        fD,max = (v/c) * fc
        
        Args:
            velocity_ms: Velocity in m/s
        Returns:
            Maximum Doppler frequency in Hz
        """
        return (velocity_ms / self.SPEED_OF_LIGHT) * self.CARRIER_FREQUENCY
    
    def get_coherence_time(self, velocity_ms):
        """
        Calculate channel coherence time (Equation 3.11)
        Tc ≈ 0.423 / fD,max
        
        Args:
            velocity_ms: Velocity in m/s
        Returns:
            Coherence time in seconds
        """
        fd_max = self.get_doppler_frequency(velocity_ms)
        if fd_max == 0:
            return float('inf')
        return 0.423 / fd_max
    
    def print_summary(self):
        """Print configuration summary"""
        print("="*60)
        print("TEMPORAL DeepMIMO DATASET GENERATION - CONFIGURATION")
        print("="*60)
        print(f"\n📡 SCENARIO: {self.SCENARIO}")
        print(f"   Carrier Frequency: {self.CARRIER_FREQUENCY/1e9:.1f} GHz")
        print(f"   Antenna Config: {self.ANTENNA_CONFIG} ({self.Nr}×{self.Nt})")
        
        print(f"\n⏱️  TEMPORAL PARAMETERS:")
        print(f"   Sampling Interval (Δt): {self.SAMPLING_INTERVAL*1e6:.2f} μs")
        print(f"   Total Time Steps: {self.TOTAL_TIME_STEPS}")
        print(f"   Sequence Length (T): {self.SEQUENCE_LENGTH}")
        
        print(f"\n🚗 VELOCITY CLASSES:")
        for name, vel_kmh in self.VELOCITY_CLASSES.items():
            vel_ms = self.kmh_to_ms(vel_kmh)
            fd = self.get_doppler_frequency(vel_ms)
            tc = self.get_coherence_time(vel_ms)
            print(f"   {name:20s}: {vel_kmh:6.1f} km/h | "
                  f"fD={fd:7.1f} Hz | Tc={tc*1e3:6.2f} ms")
        
        print(f"\n📊 DATASET SIZE:")
        sequences_per_user = (self.TOTAL_TIME_STEPS - self.SEQUENCE_LENGTH - 
                              self.PREDICTION_HORIZON + 1) // self.STRIDE
        total_sequences = (len(self.VELOCITY_CLASSES) * 
                          self.NUM_USERS_PER_VELOCITY * sequences_per_user)
        print(f"   Users per velocity: {self.NUM_USERS_PER_VELOCITY}")
        print(f"   Sequences per user: {sequences_per_user}")
        print(f"   Total sequences: {total_sequences:,}")
        
        print(f"\n🔊 CHANNEL & NOISE:")
        print(f"   AWGN: {'✓' if self.CHANNEL_CONFIG['enable_awgn'] else '✗'}")
        print(f"   Rayleigh: {'✓' if self.CHANNEL_CONFIG['enable_rayleigh'] else '✗'}")
        print(f"   Rician: {'✓' if self.CHANNEL_CONFIG['enable_rician'] else '✗'}")
        print(f"   Default SNR: {self.DEFAULT_SNR_DB} dB")
        print(f"   SNR-Velocity Coupling: {'✓' if self.SNR_VELOCITY_COUPLING else '✗'}")
        print("="*60 + "\n")


# Create global config instance
config = Config()

if __name__ == "__main__":
    # Test configuration
    config.print_summary()