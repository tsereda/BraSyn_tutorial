#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def load_nifti(path):
    """Load and return nifti data and the nifti object"""
    nii = nib.load(str(path))
    return nii.get_fdata(), nii

def swap_labels(segmentation, label1=3, label2=4):
    """Swap two labels in segmentation"""
    seg_copy = segmentation.copy()
    
    # Create masks for each label
    mask1 = segmentation == label1
    mask2 = segmentation == label2
    
    # Swap the labels
    seg_copy[mask1] = label2
    seg_copy[mask2] = label1
    
    return seg_copy

def analyze_labels(segmentation):
    """Analyze label distribution in segmentation"""
    unique_labels, counts = np.unique(segmentation, return_counts=True)
    print("\nLabel distribution:")
    for label, count in zip(unique_labels, counts):
        percentage = (count / segmentation.size) * 100
        print(f"  Label {int(label)}: {count:,} voxels ({percentage:.2f}%)")
    return unique_labels, counts

def create_comparison_plot(original_seg, swapped_seg, slice_idx=None):
    """Create before/after comparison plot"""
    if slice_idx is None:
        slice_idx = original_seg.shape[2] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original segmentation
    orig_slice = original_seg[:, :, slice_idx]
    axes[0, 0].imshow(orig_slice, cmap='jet', vmin=0, vmax=4)
    axes[0, 0].set_title('Original - Full Segmentation')
    axes[0, 0].axis('off')
    
    # Original - only label 3
    orig_label3 = (orig_slice == 3).astype(int)
    axes[0, 1].imshow(orig_label3, cmap='Reds', vmin=0, vmax=1)
    axes[0, 1].set_title('Original - Label 3 (Red)')
    axes[0, 1].axis('off')
    
    # Original - only label 4
    orig_label4 = (orig_slice == 4).astype(int)
    axes[0, 2].imshow(orig_label4, cmap='Blues', vmin=0, vmax=1)
    axes[0, 2].set_title('Original - Label 4 (Blue)')
    axes[0, 2].axis('off')
    
    # Swapped segmentation
    swap_slice = swapped_seg[:, :, slice_idx]
    axes[1, 0].imshow(swap_slice, cmap='jet', vmin=0, vmax=4)
    axes[1, 0].set_title('Swapped - Full Segmentation')
    axes[1, 0].axis('off')
    
    # Swapped - only label 3
    swap_label3 = (swap_slice == 3).astype(int)
    axes[1, 1].imshow(swap_label3, cmap='Reds', vmin=0, vmax=1)
    axes[1, 1].set_title('Swapped - Label 3 (Red)')
    axes[1, 1].axis('off')
    
    # Swapped - only label 4
    swap_label4 = (swap_slice == 4).astype(int)
    axes[1, 2].imshow(swap_label4, cmap='Blues', vmin=0, vmax=1)
    axes[1, 2].set_title('Swapped - Label 4 (Blue)')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Label Swap Comparison (Slice {slice_idx})\nTop: Original, Bottom: After swapping labels 3↔4')
    plt.tight_layout()
    return fig

def main():
    print("BraTS Label Swap Tool")
    print("This script swaps labels 3 and 4 in segmentation files")
    print("=" * 60)
    
    # Find prediction files
    possible_pred_paths = [
        "./outputs/*.nii.gz",
        "../outputs/*.nii.gz", 
        "../brats-synthesis/outputs/*.nii.gz"
    ]
    
    pred_files = []
    for pattern in possible_pred_paths:
        pred_files = glob.glob(pattern)
        if pred_files:
            print(f"Found {len(pred_files)} prediction files at {pattern}")
            break
    
    if not pred_files:
        print("No prediction files found!")
        return
    
    # Use the first file as test
    test_file = pred_files[0]
    print(f"\nProcessing test file: {Path(test_file).name}")
    
    # Load the segmentation
    original_seg, nii_obj = load_nifti(test_file)
    
    print(f"Segmentation shape: {original_seg.shape}")
    print(f"Data type: {original_seg.dtype}")
    
    # Analyze original labels
    print("\n" + "="*40)
    print("ORIGINAL SEGMENTATION:")
    analyze_labels(original_seg)
    
    # Swap labels 3 and 4
    swapped_seg = swap_labels(original_seg, label1=3, label2=4)
    
    # Analyze swapped labels
    print("\n" + "="*40)
    print("AFTER SWAPPING LABELS 3 ↔ 4:")
    analyze_labels(swapped_seg)
    
    # Create comparison visualization
    fig = create_comparison_plot(original_seg, swapped_seg)
    
    # Save the plot
    plot_filename = f"label_swap_comparison_{Path(test_file).stem}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved as: {plot_filename}")
    
    # Save the corrected segmentation
    corrected_filename = f"corrected_{Path(test_file).name}"
    
    # Create new NIfTI file with corrected labels
    corrected_nii = nib.Nifti1Image(swapped_seg.astype(original_seg.dtype), 
                                   nii_obj.affine, 
                                   nii_obj.header)
    
    nib.save(corrected_nii, corrected_filename)
    print(f"Corrected segmentation saved as: {corrected_filename}")
    
    # Show the plot
    plt.show()
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("- Labels 3 and 4 have been swapped")
    print("- Label 3 should now represent: Enhancing Tumor (ET)")
    print("- Label 4 should now represent: Whole Tumor region (WT)")
    print("- Use this corrected file to verify the swap worked as expected")
    
    # Ask if user wants to process all files
    response = input("\nDo you want to swap labels 3↔4 for ALL prediction files? (y/n): ")
    if response.lower() == 'y':
        print(f"\nProcessing all {len(pred_files)} files...")
        
        corrected_dir = Path("corrected_outputs")
        corrected_dir.mkdir(exist_ok=True)
        
        for i, pred_file in enumerate(pred_files):
            print(f"Processing {i+1}/{len(pred_files)}: {Path(pred_file).name}")
            
            # Load segmentation
            seg_data, nii_obj = load_nifti(pred_file)
            
            # Swap labels
            corrected_seg = swap_labels(seg_data, label1=3, label2=4)
            
            # Save corrected version
            output_path = corrected_dir / Path(pred_file).name
            corrected_nii = nib.Nifti1Image(corrected_seg.astype(seg_data.dtype), 
                                           nii_obj.affine, 
                                           nii_obj.header)
            nib.save(corrected_nii, output_path)
        
        print(f"\nAll files processed and saved to: {corrected_dir}/")
        print("You can now use these corrected files for evaluation!")

if __name__ == "__main__":
    main()