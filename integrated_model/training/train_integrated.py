"""
Training Script for Integrated Model

Trains the integrated model combining:
1. Silicon-Menagerie pretrained ViT (frozen backbone)
2. STEVE-style slot attention (trainable)
3. SlotContrast temporal consistency (trainable)

Usage:
    python train_integrated.py integrated_saycam.yml --data-dir /path/to/data --log-dir ./logs
"""

import sys
import os
from pathlib import Path

# Add SlotContrast to path
SLOTCONTRAST_PATH = Path(__file__).parent.parent.parent / 'slotcontrast'
sys.path.insert(0, str(SLOTCONTRAST_PATH))

# Import SlotContrast training infrastructure
from slotcontrast.train import main, parser


def setup_custom_modules():
    """
    Register custom modules with SlotContrast framework

    This allows the config to use our custom encoder and other modules
    """
    # Import custom modules
    sys.path.insert(0, str(Path(__file__).parent))
    from custom_encoders import SiliconMenagerieEncoder

    # You can register custom modules here if needed
    # For now, we'll use the existing SlotContrast infrastructure
    # with TimmExtractor that can load the silicon-menagerie models

    print("Custom modules ready")


if __name__ == '__main__':
    # Setup custom modules
    setup_custom_modules()

    # Parse arguments
    args = parser.parse_args()

    # Run training using SlotContrast's main function
    print("=" * 80)
    print("INTEGRATED MODEL TRAINING")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Log directory: {args.log_dir}")
    print("=" * 80)

    result = main(args)

    if result == 0:
        print("\nTraining completed successfully!")
    elif result == 1:
        print("\nTraining stopped due to timeout")
    else:
        print(f"\nTraining finished with code: {result}")
