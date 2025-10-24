#!/usr/bin/env python3
"""
Training Script for Integrated Model V2

Combines SlotContrast + STEVE + Silicon-Menagerie

Usage:
    python train_integrated_v2.py configs/integrated_v2_saycam.yml --data-dir /path/to/data

This script automatically uses the extended model builder that includes STEVE components.
"""

import sys
import argparse
from pathlib import Path

# Import SlotContrast training infrastructure
from slotcontrast import train as slotcontrast_train
from slotcontrast import configuration, data, metrics

# Import extended model builder
import slotcontrast.models_steve as models_steve


def main():
    """Main training function"""
    # Parse arguments using SlotContrast's parser
    args = slotcontrast_train.parser.parse_args()

    print("=" * 80)
    print("INTEGRATED MODEL V2 TRAINING")
    print("SlotContrast + STEVE + Silicon-Menagerie")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Log directory: {args.log_dir}")
    print("=" * 80)
    print()

    # Modify the models module to use our extended builder
    original_build = slotcontrast_train.models.build
    slotcontrast_train.models.build = models_steve.build

    # Register silicon-menagerie ViT if available
    try:
        from slotcontrast.modules import silicon_vit
        print("Silicon-menagerie ViT backbone registered")
    except ImportError as e:
        print(f"Warning: Could not import silicon_vit: {e}")
        print("Will fall back to standard encoders if silicon ViT is requested")

    # Run training using SlotContrast's infrastructure
    result = slotcontrast_train.main(args)

    # Restore original builder
    slotcontrast_train.models.build = original_build

    if result == slotcontrast_train.RESULT_FINISHED:
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
    elif result == slotcontrast_train.RESULT_TIMEOUT:
        print("\n" + "=" * 80)
        print("Training stopped due to timeout")
        print("=" * 80)
    else:
        print(f"\nTraining finished with code: {result}")

    return result


if __name__ == "__main__":
    sys.exit(main())
