#!/usr/bin/env python3
"""
This script creates a pseudo validation set during the validation stage and a test set during the final evaluation.
It randomly drops one modality from each case to simulate missing data scenarios.
"""

import os
import random
import numpy as np
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Create a pseudo validation set by randomly dropping one modality from each case'
    )
    parser.add_argument(
        'source_folder',
        help='Path to the source validation set folder'
    )
    parser.add_argument(
        'output_folder', 
        help='Path to the output folder for the pseudo validation set'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=123456,
        help='Random seed for reproducibility (default: 123456)'
    )
    parser.add_argument(
        '--modalities',
        nargs='+',
        default=['t1c', 't1n', 't2f', 't2w'],
        help='List of modalities (default: t1c t1n t2f t2w)'
    )
    
    args = parser.parse_args()
    
    val_set_folder = args.source_folder
    val_set_missing = args.output_folder
    
    # Validate source folder exists
    if not os.path.exists(val_set_folder):
        print(f"Error: Source folder '{val_set_folder}' does not exist!")
        return 1
    
    # Create output folder if it doesn't exist
    if not os.path.exists(val_set_missing):
        os.makedirs(val_set_missing)
        print(f"Created output folder: {val_set_missing}")
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    modality_list = args.modalities
    
    # Get list of folders/cases
    folder_list = os.listdir(val_set_folder)
    folder_list.sort()
    
    if not folder_list:
        print(f"Warning: No folders found in {val_set_folder}")
        return 1
    
    # Generate random indices for which modality to drop for each case
    drop_index = np.random.randint(0, len(modality_list), size=len(folder_list))
    
    print(f"Processing {len(folder_list)} cases...")
    print(f"Modalities: {modality_list}")
    print(f"Random seed: {args.seed}")
    print("-" * 50)
    
    for count, ff in enumerate(folder_list):
        case_output_path = os.path.join(val_set_missing, ff)
        if not os.path.exists(case_output_path):
            os.makedirs(case_output_path)
        
        case_source_path = os.path.join(val_set_folder, ff)
        file_list = os.listdir(case_source_path)
        
        dropped_modality = modality_list[drop_index[count]]
        copied_files = 0
        
        for mm in file_list:
            # Skip files that contain the dropped modality
            if dropped_modality not in mm:
                source_file = os.path.join(case_source_path, mm)
                dest_file = os.path.join(case_output_path, mm)
                shutil.copyfile(source_file, dest_file)
                copied_files += 1
        
        print(f"Case {ff}: dropped '{dropped_modality}', copied {copied_files} files")
    
    print("-" * 50)
    print(f"Pseudo validation set created successfully in: {val_set_missing}")
    return 0

if __name__ == "__main__":
    exit(main())