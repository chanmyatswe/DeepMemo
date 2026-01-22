"""
Main Dataset Generation Pipeline
Combines DeepMIMO, Doppler simulation, and channel models
"""

import numpy as np
import os
import pickle
from datetime import datetime
from tqdm import tqdm

try:
    import DeepMIMOv3 as DeepMIMO
except ImportError:
    print("Warning: DeepMIMO not installed. Install with: pip install DeepMIMO")

from doppler_simulator import DopplerSimulator
from channel_models import ChannelModels


class TemporalDatasetGenerator:
    """
    Complete pipeline for temporal CSI dataset generation
    
    Workflow:
    1. Load DeepMIMO static snapshots
    2. Apply Doppler effects (Clarke model)
    3. Add channel impairments (AWGN, Rayleigh, Rician)
    4. Create sliding window sequences
    5. Split into train/val/test
    6. Save processed dataset
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self.doppler_sim = DopplerSimulator(config)
        self.channel_model = ChannelModels(config)
        
        self.deepmimo_dataset = None
        self.temporal_data = {}  # Store data for each velocity class
        
    def load_deepmimo_dataset(self):
        """
        Load and configure DeepMIMO dataset
        
        Returns:
            DeepMIMO dataset object
        """
        print("\n" + "="*60)
        print("LOADING DeepMIMO DATASET")
        print("="*60)
        
        # Configure DeepMIMO parameters
        parameters = DeepMIMO.default_params()
        
        # Scenario
        parameters['scenario'] = self.config.SCENARIO
        print(f"Scenario: {self.config.SCENARIO}")
        
        # Active base stations
        parameters['active_BS'] = np.array(self.config.ACTIVE_BS)
        
        # User grid
        parameters['user_row_first'] = self.config.USER_ROW_FIRST
        parameters['user_row_last'] = self.config.USER_ROW_LAST
        parameters['user_row_subsampling'] = self.config.USER_ROW_SUBSAMPLING
        
        # OFDM parameters
        parameters['num_subcarrier'] = self.config.NUM_SUBCARRIERS
        parameters['subcarrier_spacing'] = self.config.SUBCARRIER_SPACING
        parameters['bandwidth'] = self.config.BANDWIDTH
        
        # MIMO configuration
        if self.config.ANTENNA_CONFIG == 'SISO':
            parameters['enable_BS2BS'] = False
            # SISO is default in DeepMIMO
        else:
            # For MIMO, configure antenna arrays
            parameters['num_ant_BS'] = self.config.Nt
            parameters['num_ant_UE'] = self.config.Nr
        
        print(f"Antenna Configuration: {self.config.ANTENNA_CONFIG} "
              f"({self.config.Nr}×{self.config.Nt})")
        print(f"Number of Subcarriers: {self.config.NUM_SUBCARRIERS}")
        print(f"Subcarrier Spacing: {self.config.SUBCARRIER_SPACING/1e3:.0f} kHz")
        
        # Generate dataset
        print("\nGenerating DeepMIMO dataset...")
        dataset = DeepMIMO.generate_data(parameters)
        
        num_users = len(dataset[0]['user']['channel'])
        print(f"✓ Loaded {num_users} users")
        
        # Print channel shape
        H_sample = dataset[0]['user']['channel'][0]
        print(f"✓ Channel shape: {H_sample.shape}")
        
        self.deepmimo_dataset = dataset
        print("="*60 + "\n")
        
        return dataset
    
    def generate_velocity_class(self, velocity_class, add_noise=True):
        """
        Generate temporal dataset for one velocity class
        
        Args:
            velocity_class: Name of velocity class (e.g., 'urban_60')
            add_noise: Whether to add channel impairments
            
        Returns:
            Dictionary with processed data
        """
        print(f"\n{'='*60}")
        print(f"GENERATING: {velocity_class}")
        print(f"{'='*60}")
        
        # Generate temporal CSI with Doppler
        result = self.doppler_sim.generate_temporal_dataset(
            self.deepmimo_dataset,
            velocity_class,
            num_users=self.config.NUM_USERS_PER_VELOCITY,
            show_progress=True
        )
        
        H_temporal = result['H_temporal']
        velocity_ms = result['velocity_ms']
        velocity_kmh = result['velocity_kmh']
        
        print(f"\n✓ Generated temporal CSI: {H_temporal.shape}")
        print(f"  Velocity: {velocity_kmh:.1f} km/h ({velocity_ms:.2f} m/s)")
        print(f"  Max Doppler: {result['fd_max']:.2f} Hz")
        print(f"  Coherence Time: {result['Tc']*1e3:.2f} ms")
        
        # Add channel impairments (noise, fading)
        if add_noise:
            print("\nApplying channel impairments...")
            
            # Determine SNR
            if self.config.SNR_VELOCITY_COUPLING:
                snr_db = self.config.SNR_VELOCITY_MAPPING.get(
                    velocity_class, self.config.DEFAULT_SNR_DB
                )
                print(f"  SNR (velocity-coupled): {snr_db} dB")
            else:
                snr_db = self.config.DEFAULT_SNR_DB
                print(f"  SNR: {snr_db} dB")
            
            # Apply channel effects to all users
            H_noisy = np.zeros_like(H_temporal)
            for user_idx in range(len(H_temporal)):
                H_noisy[user_idx] = self.channel_model.apply_channel_effects(
                    H_temporal[user_idx],
                    snr_db=snr_db,
                    fading_type='rayleigh'  # Can change to 'rician'
                )
            
            print(f"  ✓ Applied: {self.channel_model.get_channel_type_string()}")
            result['H_noisy'] = H_noisy
            result['snr_db'] = snr_db
        else:
            result['H_noisy'] = H_temporal.copy()
            result['snr_db'] = None
        
        # Create sliding windows
        print("\nCreating sliding window sequences...")
        X_seq, Y_seq = self.doppler_sim.create_sliding_windows(
            result['H_noisy'],
            window_size=self.config.SEQUENCE_LENGTH,
            pred_horizon=self.config.PREDICTION_HORIZON,
            stride=self.config.STRIDE
        )
        
        print(f"  ✓ Input sequences (X): {X_seq.shape}")
        print(f"  ✓ Target sequences (Y): {Y_seq.shape}")
        
        result['X_sequences'] = X_seq
        result['Y_sequences'] = Y_seq
        
        # Also store clean version (without noise) as ground truth
        if add_noise:
            X_clean, Y_clean = self.doppler_sim.create_sliding_windows(
                H_temporal,
                window_size=self.config.SEQUENCE_LENGTH,
                pred_horizon=self.config.PREDICTION_HORIZON,
                stride=self.config.STRIDE
            )
            result['X_clean'] = X_clean
            result['Y_clean'] = Y_clean
        
        print(f"{'='*60}\n")
        
        return result
    
    def generate_all_velocities(self, add_noise=True):
        """
        Generate dataset for all velocity classes
        
        Args:
            add_noise: Whether to add channel impairments
            
        Returns:
            Dictionary mapping velocity class to data
        """
        print("\n" + "="*70)
        print("GENERATING COMPLETE TEMPORAL DATASET")
        print("="*70)
        
        # Ensure DeepMIMO is loaded
        if self.deepmimo_dataset is None:
            self.load_deepmimo_dataset()
        
        # Generate for each velocity
        for velocity_class in self.config.VELOCITY_CLASSES.keys():
            data = self.generate_velocity_class(velocity_class, add_noise)
            self.temporal_data[velocity_class] = data
        
        print("\n" + "="*70)
        print("DATASET GENERATION COMPLETE")
        print("="*70)
        self._print_dataset_summary()
        
        return self.temporal_data
    
    def _print_dataset_summary(self):
        """Print summary of generated dataset"""
        total_sequences = 0
        
        print("\nDataset Summary:")
        print("-" * 70)
        print(f"{'Velocity Class':<20} {'Velocity':<15} {'Sequences':<12} {'SNR (dB)'}")
        print("-" * 70)
        
        for vel_class, data in self.temporal_data.items():
            num_seq = len(data['X_sequences'])
            total_sequences += num_seq
            vel_str = f"{data['velocity_kmh']:.0f} km/h"
            snr_str = f"{data['snr_db']}" if data['snr_db'] is not None else "N/A"
            print(f"{vel_class:<20} {vel_str:<15} {num_seq:<12} {snr_str}")
        
        print("-" * 70)
        print(f"{'TOTAL':<20} {'':<15} {total_sequences:<12}")
        print("-" * 70)
        
        # Print shape info
        if len(self.temporal_data) > 0:
            first_data = list(self.temporal_data.values())[0]
            X_shape = first_data['X_sequences'].shape
            Y_shape = first_data['Y_sequences'].shape
            print(f"\nSequence Shapes:")
            print(f"  Input (X): {X_shape}")
            print(f"  Target (Y): {Y_shape}")
            print(f"  Format: (num_sequences, time_steps, Nr, Nt, Nk)")
    
    def split_train_val_test(self, data_dict):
        """
        Split dataset into train/validation/test sets
        
        Args:
            data_dict: Dictionary with X_sequences and Y_sequences
            
        Returns:
            Dictionary with train/val/test splits
        """
        X = data_dict['X_sequences']
        Y = data_dict['Y_sequences']
        
        num_samples = len(X)
        
        # Calculate split indices
        train_end = int(num_samples * self.config.TRAIN_RATIO)
        val_end = train_end + int(num_samples * self.config.VAL_RATIO)
        
        # Split
        split_data = {
            'X_train': X[:train_end],
            'Y_train': Y[:train_end],
            'X_val': X[train_end:val_end],
            'Y_val': Y[train_end:val_end],
            'X_test': X[val_end:],
            'Y_test': Y[val_end:],
        }
        
        # Also include clean versions if available
        if 'X_clean' in data_dict:
            X_clean = data_dict['X_clean']
            Y_clean = data_dict['Y_clean']
            split_data.update({
                'X_train_clean': X_clean[:train_end],
                'Y_train_clean': Y_clean[:train_end],
                'X_val_clean': X_clean[train_end:val_end],
                'Y_val_clean': Y_clean[train_end:val_end],
                'X_test_clean': X_clean[val_end:],
                'Y_test_clean': Y_clean[val_end:],
            })
        
        return split_data
    
    # def save_dataset(self, output_name=None):
    #     """
    #     Save generated dataset to disk
        
    #     Args:
    #         output_name: Custom output filename (None = auto-generate)
    #     """
    #     if not self.temporal_data:
    #         print("Error: No data to save. Run generate_all_velocities() first.")
    #         return
        
    #     # Generate filename
    #     if output_name is None:
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         output_name = f"temporal_dataset_{self.config.ANTENNA_CONFIG}_{timestamp}.pkl"
        
    #     output_path = os.path.join(self.config.PROCESSED_DATA_DIR, output_name)
        
    #     print(f"\nSaving dataset to: {output_path}")
        
    #     # Prepare data for saving
    #     save_data = {
    #         'config': {
    #             'scenario': self.config.SCENARIO,
    #             'antenna_config': self.config.ANTENNA_CONFIG,
    #             'carrier_frequency': self.config.CARRIER_FREQUENCY,
    #             'num_subcarriers': self.config.NUM_SUBCARRIERS,
    #             'sequence_length': self.config.SEQUENCE_LENGTH,
    #             'sampling_interval': self.config.SAMPLING_INTERVAL,
    #             'velocity_classes': self.config.VELOCITY_CLASSES,
    #             'channel_config': self.config.CHANNEL_CONFIG,
    #         },
    #         'velocity_data': {}
    #     }
        
    #     # Split each velocity class and save
    #     for vel_class, data in self.temporal_data.items():
    #         split_data = self.split_train_val_test(data)
            
    #         # Store split data along with metadata
    #         save_data['velocity_data'][vel_class] = {
    #             'splits': split_data,
    #             'metadata': {
    #                 'velocity_ms': data['velocity_ms'],
    #                 'velocity_kmh': data['velocity_kmh'],
    #                 'fd_max': data['fd_max'],
    #                 'Tc': data['Tc'],
    #                 'snr_db': data['snr_db'],
    #             }
    #         }
            
    #         print(f"  ✓ {vel_class}: train={len(split_data['X_train'])}, "
    #               f"val={len(split_data['X_val'])}, test={len(split_data['X_test'])}")
        
    #     # Save with pickle
    #     with open(output_path, 'wb') as f:
    #         pickle.dump(save_data, f, protocol=4)
        
    #     file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    #     print(f"\n✓ Dataset saved successfully!")
    #     print(f"  File size: {file_size_mb:.2f} MB")
    #     print(f"  Path: {output_path}")
        
    #     return output_path
    def save_dataset(self, output_name=None, save_mode="raw"):
        """
        Save dataset.

        Option-A (your current goal): save_mode="raw"
        - Saves ONLY:
            H_temporal (clean Doppler CSI)
            H_noisy    (after channel effects; equals H_temporal if add_noise=False)
        - NO sequences, NO train/val/test splits.
        """
        if not self.temporal_data:
            print("Error: No data to save. Run generate_all_velocities() first.")
            return

        if save_mode != "raw":
            raise ValueError("For now, use save_mode='raw' (Option-A).")

        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"temporal_RAW_{self.config.ANTENNA_CONFIG}_{timestamp}.pkl"

        output_path = os.path.join(self.config.PROCESSED_DATA_DIR, output_name)
        print(f"\nSaving RAW dataset to: {output_path}")

        save_data = {
            "dataset_type": "RAW_TEMPORAL_CSI",
            "save_mode": "raw",
            "config": {
                "scenario": self.config.SCENARIO,
                "antenna_config": self.config.ANTENNA_CONFIG,
                "carrier_frequency": self.config.CARRIER_FREQUENCY,
                "num_subcarriers": self.config.NUM_SUBCARRIERS,
                "sampling_interval": self.config.SAMPLING_INTERVAL,
                "velocity_classes": self.config.VELOCITY_CLASSES,
                "channel_config": self.config.CHANNEL_CONFIG,
                "num_users_per_velocity": self.config.NUM_USERS_PER_VELOCITY,
                "total_time_steps": self.config.TOTAL_TIME_STEPS,
            },
            "velocity_data": {}
        }

        for vel_class, d in self.temporal_data.items():
            save_data["velocity_data"][vel_class] = {
                "raw": {
                    "H_temporal": d["H_temporal"],
                    "H_noisy": d["H_noisy"],
                },
                "metadata": {
                    "velocity_ms": d["velocity_ms"],
                    "velocity_kmh": d["velocity_kmh"],
                    "fd_max": d["fd_max"],
                    "Tc": d["Tc"],
                    "snr_db": d.get("snr_db", None),
                }
            }
            print(f"  ✓ {vel_class}: H_temporal={d['H_temporal'].shape}, snr_db={d.get('snr_db', None)}")

        with open(output_path, "wb") as f:
            pickle.dump(save_data, f, protocol=4)

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n✓ RAW dataset saved successfully!")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Path: {output_path}")

        return output_path

def load_dataset(filepath):
    print(f"Loading dataset from: {filepath}")
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    print("✓ Dataset loaded successfully!")

    print("\nDataset Type:", data.get("dataset_type", "UNKNOWN"))
    print("Save Mode:", data.get("save_mode", "UNKNOWN"))

    print("\nDataset Configuration:")
    for key, value in data["config"].items():
        print(f"  {key}: {value}")

    print("\nVelocity Classes Summary:")
    for vel_class, vel_data in data["velocity_data"].items():
        md = vel_data["metadata"]

        if "raw" in vel_data:
            Ht = vel_data["raw"]["H_temporal"]
            Hn = vel_data["raw"]["H_noisy"]
            print(f"  {vel_class}: {md['velocity_kmh']:.0f} km/h | "
                  f"H_temporal={Ht.shape} | H_noisy={Hn.shape} | fd_max={md['fd_max']:.2f} Hz")
        else:
            print(f"  {vel_class}: {md['velocity_kmh']:.0f} km/h | (non-raw format)")

    return data


# def load_dataset(filepath):
#     """
#     Load saved dataset
    
#     Args:
#         filepath: Path to saved .pkl file
        
#     Returns:
#         Loaded dataset dictionary
#     """
#     print(f"Loading dataset from: {filepath}")
#     with open(filepath, 'rb') as f:
#         data = pickle.load(f)
    
#     print("✓ Dataset loaded successfully!")
    
#     # Print summary
#     print("\nDataset Configuration:")
#     for key, value in data['config'].items():
#         print(f"  {key}: {value}")
    
#     print("\nVelocity Classes:")
#     for vel_class, vel_data in data['velocity_data'].items():
#         metadata = vel_data['metadata']
#         splits = vel_data['splits']
#         print(f"  {vel_class}: {metadata['velocity_kmh']:.0f} km/h, "
#               f"train={len(splits['X_train'])}")
    
#     return data


# if __name__ == "__main__":
#     from config import config
    
#     # Test dataset generation
#     print("\nStarting dataset generation test...")
    
#     generator = TemporalDatasetGenerator(config)
    
#     # Load DeepMIMO
#     generator.load_deepmimo_dataset()
    
#     # Generate for one velocity (test)
#     test_data = generator.generate_velocity_class('pedestrian_5', add_noise=True)
    
#     print("\n✓ Test successful!")
#     print(f"Generated {len(test_data['X_sequences'])} sequences")

if __name__ == "__main__":
    from config import config

    print("\nStarting RAW dataset generation test (Option-A)...")

    generator = TemporalDatasetGenerator(config)
    generator.load_deepmimo_dataset()

    # Generate one velocity only (fast debug)
    test_data = generator.generate_velocity_class("pedestrian_5", add_noise=True)
    generator.temporal_data["pedestrian_5"] = test_data

    output_path = generator.save_dataset(output_name="test_dataset_raw.pkl", save_mode="raw")
    print("\n✓ Test successful!")
    print("Saved:", output_path)
