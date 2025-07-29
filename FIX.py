#!/usr/bin/env python3
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def load_nifti(path):
    """Load and return nifti data and the nifti object"""
    nii = nib.load(str(path))
    return nii.get_fdata(), nii

def swap_labels_1_2(segmentation):
    """Swap labels 1 and 2 in segmentation"""
    seg_copy = segmentation.copy()
    
    # Create masks for each label
    mask1 = segmentation == 1
    mask2 = segmentation == 2
    
    # Swap the labels
    seg_copy[mask1] = 2
    seg_copy[mask2] = 1
    
    return seg_copy

def compute_dice_score(pred_mask, gt_mask):
    """Compute Dice score between two binary masks"""
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    
    if union > 0:
        return (2.0 * intersection) / union
    else:
        return 1.0 if np.sum(pred_mask) == 0 else 0.0

def compare_performance(original_seg, swapped_seg, gt_seg, case_name):
    """Compare original vs swapped performance"""
    print(f"\n📊 PERFORMANCE COMPARISON: {case_name}")
    print("=" * 60)
    
    # Compute Dice for both versions
    print(f"{'Label':<8} {'Original':<10} {'Swapped':<10} {'Change':<10}")
    print("-" * 50)
    
    total_orig = []
    total_swap = []
    
    for label in [1, 2, 3, 4]:
        # Original
        orig_mask = (original_seg == label).astype(float)
        gt_mask = (gt_seg == label).astype(float)
        orig_dice = compute_dice_score(orig_mask, gt_mask)
        
        # Swapped
        swap_mask = (swapped_seg == label).astype(float)
        swap_dice = compute_dice_score(swap_mask, gt_mask)
        
        change = swap_dice - orig_dice
        total_orig.append(orig_dice)
        total_swap.append(swap_dice)
        
        print(f"{label:<8} {orig_dice:.3f}     {swap_dice:.3f}     {change:+.3f}")
    
    orig_avg = np.mean(total_orig)
    swap_avg = np.mean(total_swap)
    
    print("-" * 50)
    print(f"{'AVERAGE':<8} {orig_avg:.3f}     {swap_avg:.3f}     {swap_avg-orig_avg:+.3f}")
    
    if swap_avg > orig_avg:
        print(f"✅ IMPROVED by {swap_avg-orig_avg:.3f}!")
        return True
    else:
        print(f"❌ Decreased by {orig_avg-swap_avg:.3f}")
        return False

def create_before_after_plot(original_seg, swapped_seg, gt_seg, case_name, slice_idx=None):
    """Create before/after comparison plot"""
    if slice_idx is None:
        slice_idx = original_seg.shape[2] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original
    orig_slice = original_seg[:, :, slice_idx]
    axes[0].imshow(orig_slice, cmap='jet', vmin=0, vmax=4)
    axes[0].set_title('Original Prediction')
    axes[0].axis('off')
    
    # Swapped
    swap_slice = swapped_seg[:, :, slice_idx]
    axes[1].imshow(swap_slice, cmap='jet', vmin=0, vmax=4)
    axes[1].set_title('After 1↔2 Swap')
    axes[1].axis('off')
    
    # Ground Truth
    gt_slice = gt_seg[:, :, slice_idx]
    axes[2].imshow(gt_slice, cmap='jet', vmin=0, vmax=4)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    plt.suptitle(f'Label 1↔2 Swap Comparison: {case_name} (Slice {slice_idx})')
    plt.tight_layout()
    
    # Save the plot
    plot_file = f"swap_1_2_comparison_{case_name}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📸 Comparison saved as: {plot_file}")
    
    return fig

def main():
    print("🔄 BraTS Label 1↔2 Swap Tool")
    print("=" * 50)
    
    # Find prediction files
    pred_files = glob.glob("outputs/*.nii.gz")
    
    if not pred_files:
        print("❌ No prediction files found in outputs/")
        return
    
    print(f"Found {len(pred_files)} prediction files")
    
    # Check training data
    training_data_dir = Path("ASNR-MICCAI-BraTS2023-GLI-MET-TrainingData")
    if not training_data_dir.exists():
        print(f"❌ Training data directory not found: {training_data_dir}")
        return
    
    # Test with first file
    test_file = pred_files[0]
    case_name = Path(test_file).name.replace('.nii.gz', '').replace('.nii', '')
    
    print(f"\n🧪 TESTING with: {case_name}")
    
    # Find ground truth
    case_dir = training_data_dir / case_name
    gt_file = case_dir / f"{case_name}-seg.nii.gz"
    
    if not gt_file.exists():
        print(f"❌ Ground truth not found: {gt_file}")
        return
    
    # Load files
    original_seg, nii_obj = load_nifti(test_file)
    gt_seg = load_nifti(gt_file)
    
    print(f"Original shape: {original_seg.shape}")
    
    # Perform 1↔2 swap
    swapped_seg = swap_labels_1_2(original_seg)
    
    # Compare performance
    improved = compare_performance(original_seg, swapped_seg, gt_seg, case_name)
    
    # Create visualization
    create_before_after_plot(original_seg, swapped_seg, gt_seg, case_name)
    
    if improved:
        print(f"\n✅ The 1↔2 swap IMPROVED performance!")
        response = input("\nProcess all files with 1↔2 swap? (y/n): ")
        
        if response.lower() == 'y':
            # Create output directory
            output_dir = Path("fixed_outputs")
            output_dir.mkdir(exist_ok=True)
            
            print(f"\n🔄 Processing all {len(pred_files)} files...")
            
            total_improvements = 0
            
            for i, pred_file in enumerate(pred_files):
                case_name = Path(pred_file).name.replace('.nii.gz', '').replace('.nii', '')
                print(f"Processing {i+1}/{len(pred_files)}: {case_name}")
                
                # Load original
                seg_data, nii_obj = load_nifti(pred_file)
                
                # Swap labels 1↔2
                fixed_seg = swap_labels_1_2(seg_data)
                
                # Save fixed version
                output_path = output_dir / Path(pred_file).name
                fixed_nii = nib.Nifti1Image(fixed_seg.astype(seg_data.dtype), 
                                           nii_obj.affine, 
                                           nii_obj.header)
                nib.save(fixed_nii, output_path)
                
                # Quick check if this case improved
                case_dir = training_data_dir / case_name
                gt_file = case_dir / f"{case_name}-seg.nii.gz"
                
                if gt_file.exists():
                    gt_seg = load_nifti(gt_file)
                    
                    # Quick dice comparison
                    orig_dice = []
                    swap_dice = []
                    
                    for label in [1, 2, 3, 4]:
                        orig_mask = (seg_data == label).astype(float)
                        swap_mask = (fixed_seg == label).astype(float)
                        gt_mask = (gt_seg == label).astype(float)
                        
                        orig_dice.append(compute_dice_score(orig_mask, gt_mask))
                        swap_dice.append(compute_dice_score(swap_mask, gt_mask))
                    
                    if np.mean(swap_dice) > np.mean(orig_dice):
                        total_improvements += 1
                        print(f"  ✅ Improved!")
                    else:
                        print(f"  ➡️  No change")
            
            print(f"\n🎉 COMPLETED!")
            print(f"📁 Fixed files saved to: {output_dir}/")
            print(f"📈 {total_improvements}/{len(pred_files)} cases improved")
            print(f"🎯 Use these fixed files for your final evaluation!")
            
    else:
        print(f"\n❌ The 1↔2 swap didn't help.")
        print(f"💡 Your original predictions might already be optimal!")
        print(f"   Consider using the original outputs/ files as-is.")

if __name__ == "__main__":
    main()