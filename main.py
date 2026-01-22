"""
Main Entry Point for Temporal DeepMIMO Dataset Generation
Run this script to generate the complete dataset
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import config
from src.dataset_generator import TemporalDatasetGenerator
from src.visualizer import DatasetVisualizer


def main():
    """Main execution pipeline"""
    
    # Print configuration
    config.print_summary()
    
    # Ask user for confirmation
    print("\n" + "="*70)
    print("READY TO GENERATE DATASET")
    print("="*70)
    print("\nThis will:")
    print("  1. Download/load DeepMIMO O1_3p5 scenario")
    print("  2. Apply Doppler effects for each velocity class")
    print("  3. Add channel impairments (AWGN, Rayleigh fading)")
    print("  4. Create sliding window sequences")
    print("  5. Split into train/val/test sets")
    print("  6. Save processed dataset")
    print("  7. Generate visualization plots")
    
    # Estimate dataset size
    sequences_per_user = (config.TOTAL_TIME_STEPS - config.SEQUENCE_LENGTH - 
                          config.PREDICTION_HORIZON + 1) // config.STRIDE
    total_sequences = (len(config.VELOCITY_CLASSES) * 
                      config.NUM_USERS_PER_VELOCITY * sequences_per_user)
    
    print(f"\nEstimated dataset size: {total_sequences:,} sequences")
    print(f"Estimated time: ~5-15 minutes (depending on your system)")
    
    response = input("\nProceed? (y/n): ").lower().strip()
    if response != 'y':
        print("Aborted.")
        return
    
    print("\n" + "="*70)
    print("STARTING DATASET GENERATION")
    print("="*70 + "\n")
    
    # Initialize generator
    generator = TemporalDatasetGenerator(config)
    
    # Step 1: Load DeepMIMO
    try:
        generator.load_deepmimo_dataset()
    except Exception as e:
        print(f"\n❌ Error loading DeepMIMO: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection (DeepMIMO downloads data)")
        print("  2. Verify scenario name is correct: 'O1_3p5'")
        print("  3. Install DeepMIMO: pip install DeepMIMO")
        return
    
    # Step 2: Generate temporal dataset for all velocities
    try:
        generator.generate_all_velocities(add_noise=True)
    except Exception as e:
        print(f"\n❌ Error generating dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Save dataset
    try:
        output_path = generator.save_dataset()
    except Exception as e:
        print(f"\n❌ Error saving dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    try:
        visualizer = DatasetVisualizer(config)
        visualizer.visualize_dataset(generator.temporal_data)
        print("\n✓ Visualizations saved to:", config.VIZ_DIR)
    except Exception as e:
        print(f"\n⚠️  Warning: Could not generate visualizations: {e}")
    
    # Done!
    print("\n" + "="*70)
    print("✅ DATASET GENERATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Dataset saved to: {output_path}")
    print(f"📊 Visualizations: {config.VIZ_DIR}")
    print(f"📋 Logs: {config.LOG_DIR}")
    
    print("\n🎯 Next Steps:")
    print("  1. Explore the dataset in Jupyter notebook")
    print("  2. Implement CNN-LSTM model for training")
    print("  3. Compare with classical methods (LS, LMMSE)")
    
    print("\n" + "="*70 + "\n")


def quick_test():
    """Quick test with minimal data for debugging"""
    print("\n🧪 RUNNING QUICK TEST MODE")
    print("="*70)
    
    # Override config for quick test
    config.NUM_USERS_PER_VELOCITY = 5  # Only 5 users
    config.TOTAL_TIME_STEPS = 50       # Only 50 time steps
    config.VELOCITY_CLASSES = {
        'pedestrian_5': 5.0,
        'urban_60': 60.0,
    }  # Only 2 velocities
    
    config.print_summary()
    
    generator = TemporalDatasetGenerator(config)
    generator.load_deepmimo_dataset()
    generator.generate_all_velocities(add_noise=True)
    
    output_path = generator.save_dataset(output_name="test_dataset.pkl")
    
    print("\n✅ Quick test successful!")
    print(f"Test dataset saved to: {output_path}")


def visualize_only():
    """Only generate visualizations from existing dataset"""
    print("\n📊 VISUALIZATION MODE")
    print("="*70)
    
    # List available datasets
    processed_dir = config.PROCESSED_DATA_DIR
    datasets = [f for f in os.listdir(processed_dir) if f.endswith('.pkl')]
    
    if not datasets:
        print(f"No datasets found in {processed_dir}")
        print("Run main() first to generate a dataset.")
        return
    
    print("\nAvailable datasets:")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset}")
    
    choice = input("\nSelect dataset number (or press Enter for latest): ").strip()
    
    if choice:
        dataset_file = datasets[int(choice) - 1]
    else:
        dataset_file = sorted(datasets)[-1]  # Latest
    
    print(f"\nLoading: {dataset_file}")
    
    from dataset_generator import load_dataset
    data = load_dataset(os.path.join(processed_dir, dataset_file))
    
    # Extract temporal data for visualization
    temporal_data = {}
    for vel_class, vel_data in data['velocity_data'].items():
        # Reconstruct format expected by visualizer
        temporal_data[vel_class] = {
            'X_sequences': vel_data['splits']['X_train'],
            'Y_sequences': vel_data['splits']['Y_train'],
            'velocity_kmh': vel_data['metadata']['velocity_kmh'],
            'velocity_ms': vel_data['metadata']['velocity_ms'],
            'fd_max': vel_data['metadata']['fd_max'],
            'Tc': vel_data['metadata']['Tc'],
        }
    
    visualizer = DatasetVisualizer(config)
    visualizer.visualize_dataset(temporal_data)
    
    print(f"\n✓ Visualizations saved to: {config.VIZ_DIR}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate Temporal DeepMIMO Dataset with Doppler Effects'
    )
    parser.add_argument(
        '--mode',
        choices=['full', 'test', 'visualize'],
        default='full',
        help='Execution mode: full generation, quick test, or visualize only'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        main()
    elif args.mode == 'test':
        quick_test()
    elif args.mode == 'visualize':
        visualize_only()