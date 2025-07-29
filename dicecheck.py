#!/usr/bin/env python3
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

def load_nifti(path):
    """Load and return nifti data"""
    nii = nib.load(str(path))
    return nii.get_fdata()

def create_three_way_comparison(original_pred, swapped_pred, ground_truth, slice_idx=None):
    """Create three-way comparison: original prediction vs swapped vs ground truth"""
    if slice_idx is None:
        slice_idx = original_pred.shape[2] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original prediction
    orig_slice = original_pred[:, :, slice_idx]
    axes[0].imshow(orig_slice, cmap='jet', vmin=0, vmax=4)
    axes[0].set_title('Original Prediction')
    axes[0].axis('off')
    
    # Swapped prediction  
    swap_slice = swapped_pred[:, :, slice_idx]
    axes[1].imshow(swap_slice, cmap='jet', vmin=0, vmax=4)
    axes[1].set_title('Swapped Prediction (2↔4)')
    axes[1].axis('off')
    
    # Ground truth
    gt_slice = ground_truth[:, :, slice_idx]
    axes[2].imshow(gt_slice, cmap='jet', vmin=0, vmax=4)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    plt.suptitle(f'Three-Way Comparison (Slice {slice_idx})')
    plt.tight_layout()
    return fig

def compute_dice_per_label(pred, gt):
    """Compute Dice score for each label"""
    dice_scores = {}
    
    for label in [1, 2, 3, 4]:  # Skip background (0)
        pred_mask = (pred == label).astype(float)
        gt_mask = (gt == label).astype(float)
        
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask)
        
        if union > 0:
            dice = (2.0 * intersection) / union
        else:
            dice = 1.0 if np.sum(pred_mask) == 0 else 0.0
            
        dice_scores[f'Label_{label}'] = dice
    
    return dice_scores

def main():
    print("Label Swap Verification Tool")
    print("=" * 50)
    
    # You'll need to specify these paths
    original_pred_file = input("Path to original prediction file: ").strip()
    swapped_pred_file = input("Path to swapped prediction file: ").strip()
    
    # Find ground truth - assuming BraTS structure
    case_name = Path(original_pred_file).name.replace('.nii.gz', '').replace('.nii', '')
    
    possible_gt_paths = [
        f"ASNR-MICCAI-BraTS2023-GLI-MET-TrainingData/{case_name}/{case_name}-seg.nii.gz",
        f"../BraSyn_tutorial/ASNR-MICCAI-BraTS2023-GLI-MET-TrainingData/{case_name}/{case_name}-seg.nii.gz"
    ]
    
    gt_file = None
    for path in possible_gt_paths:
        if Path(path).exists():
            gt_file = path
            break
    
    if gt_file is None:
        gt_file = input("Path to ground truth segmentation: ").strip()
    
    print(f"\nLoading files...")
    print(f"Original: {original_pred_file}")
    print(f"Swapped: {swapped_pred_file}")  
    print(f"Ground Truth: {gt_file}")
    
    # Load all three segmentations
    original_pred = load_nifti(original_pred_file)
    swapped_pred = load_nifti(swapped_pred_file)
    ground_truth = load_nifti(gt_file)
    
    # Compute Dice scores
    print("\n" + "="*50)
    print("DICE SCORES COMPARISON:")
    print("="*50)
    
    original_dice = compute_dice_per_label(original_pred, ground_truth)
    swapped_dice = compute_dice_per_label(swapped_pred, ground_truth)
    
    print(f"{'Label':<10} {'Original':<10} {'Swapped':<10} {'Improvement':<12}")
    print("-" * 50)
    
    for label in ['Label_1', 'Label_2', 'Label_3', 'Label_4']:
        orig_score = original_dice[label]
        swap_score = swapped_dice[label]
        improvement = swap_score - orig_score
        
        print(f"{label:<10} {orig_score:.3f}     {swap_score:.3f}     {improvement:+.3f}")
    
    # Overall average
    orig_avg = np.mean(list(original_dice.values()))
    swap_avg = np.mean(list(swapped_dice.values()))
    
    print("-" * 50)
    print(f"{'Average':<10} {orig_avg:.3f}     {swap_avg:.3f}     {swap_avg-orig_avg:+.3f}")
    
    # Create visualization
    fig = create_three_way_comparison(original_pred, swapped_pred, ground_truth)
    
    # Save the comparison
    comparison_file = f"three_way_comparison_{case_name}.png"
    plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
    print(f"\nComparison saved as: {comparison_file}")
    
    plt.show()
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    if swap_avg > orig_avg:
        print(f"✅ SWAP IMPROVED PERFORMANCE! (+{swap_avg-orig_avg:.3f} average Dice)")
        print("The label swap was successful and should be applied to all files.")
    else:
        print(f"❌ Swap decreased performance ({swap_avg-orig_avg:.3f} average Dice)")
        print("Consider trying a different label swap or investigating further.")

if __name__ == "__main__":
    main()