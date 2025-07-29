#!/usr/bin/env python3
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def load_nifti(path):
    """Load and return nifti data"""
    nii = nib.load(str(path))
    return nii.get_fdata()

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

def create_comparison_plot(original_pred, swapped_pred, ground_truth, case_name, slice_idx=None):
    """Create three-way comparison plot"""
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
    
    plt.suptitle(f'Verification: {case_name} (Slice {slice_idx})')
    plt.tight_layout()
    
    # Save the plot
    plot_file = f"verification_{case_name}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved as: {plot_file}")
    
    return fig

def main():
    print("Auto-Finding Files for Verification")
    print("=" * 50)
    
    # Find files automatically
    original_files = glob.glob("outputs/*.nii.gz")
    corrected_files = glob.glob("corrected_outputs/*.nii.gz")
    
    if not original_files:
        print("❌ No original files found in outputs/")
        return
        
    if not corrected_files:
        print("❌ No corrected files found in corrected_outputs/")
        return
    
    print(f"Found {len(original_files)} original files")
    print(f"Found {len(corrected_files)} corrected files")
    
    # Check if we have the training data
    training_data_dir = Path("ASNR-MICCAI-BraTS2023-GLI-MET-TrainingData")
    if not training_data_dir.exists():
        print(f"❌ Training data directory not found: {training_data_dir}")
        return
    
    print(f"✅ Found training data directory")
    
    # Process each case
    total_orig_dice = []
    total_swap_dice = []
    
    for original_file in original_files:
        case_name = Path(original_file).name.replace('.nii.gz', '').replace('.nii', '')
        
        # Find corresponding corrected file
        corrected_file = f"corrected_outputs/{Path(original_file).name}"
        if not Path(corrected_file).exists():
            print(f"❌ Corrected file not found for {case_name}")
            continue
            
        # Find ground truth file
        case_dir = training_data_dir / case_name
        if not case_dir.exists():
            print(f"❌ Case directory not found: {case_dir}")
            continue
            
        gt_file = case_dir / f"{case_name}-seg.nii.gz"
        if not gt_file.exists():
            print(f"❌ Ground truth not found: {gt_file}")
            continue
        
        print(f"\n🔍 Processing: {case_name}")
        print(f"  Original: {original_file}")
        print(f"  Corrected: {corrected_file}")
        print(f"  Ground Truth: {gt_file}")
        
        # Load all three files
        try:
            original_pred = load_nifti(original_file)
            swapped_pred = load_nifti(corrected_file)
            ground_truth = load_nifti(gt_file)
        except Exception as e:
            print(f"  ❌ Error loading files: {e}")
            continue
        
        # Compute Dice scores
        original_dice = compute_dice_per_label(original_pred, ground_truth)
        swapped_dice = compute_dice_per_label(swapped_pred, ground_truth)
        
        print(f"\n  📊 DICE SCORES for {case_name}:")
        print(f"  {'Label':<10} {'Original':<10} {'Swapped':<10} {'Change':<10}")
        print(f"  {'-'*45}")
        
        case_orig_scores = []
        case_swap_scores = []
        
        for label in ['Label_1', 'Label_2', 'Label_3', 'Label_4']:
            orig_score = original_dice[label]
            swap_score = swapped_dice[label]
            change = swap_score - orig_score
            
            case_orig_scores.append(orig_score)
            case_swap_scores.append(swap_score)
            
            print(f"  {label:<10} {orig_score:.3f}     {swap_score:.3f}     {change:+.3f}")
        
        # Case average
        case_orig_avg = np.mean(case_orig_scores)
        case_swap_avg = np.mean(case_swap_scores)
        
        total_orig_dice.extend(case_orig_scores)
        total_swap_dice.extend(case_swap_scores)
        
        print(f"  {'-'*45}")
        print(f"  {'Average':<10} {case_orig_avg:.3f}     {case_swap_avg:.3f}     {case_swap_avg-case_orig_avg:+.3f}")
        
        if case_swap_avg > case_orig_avg:
            print(f"  ✅ IMPROVED by {case_swap_avg-case_orig_avg:.3f}")
        else:
            print(f"  ❌ DECREASED by {case_orig_avg-case_swap_avg:.3f}")
        
        # Create visualization for this case
        create_comparison_plot(original_pred, swapped_pred, ground_truth, case_name)
    
    # Overall summary
    if total_orig_dice and total_swap_dice:
        overall_orig_avg = np.mean(total_orig_dice)
        overall_swap_avg = np.mean(total_swap_dice)
        
        print(f"\n" + "="*60)
        print(f"🎯 OVERALL RESULTS:")
        print(f"="*60)
        print(f"Original Average Dice: {overall_orig_avg:.3f}")
        print(f"Swapped Average Dice:  {overall_swap_avg:.3f}")
        print(f"Overall Change:        {overall_swap_avg-overall_orig_avg:+.3f}")
        
        if overall_swap_avg > overall_orig_avg:
            print(f"\n✅ SUCCESS! The 2↔4 label swap IMPROVED performance!")
            print(f"   Use the files in corrected_outputs/ for final evaluation")
        else:
            print(f"\n❌ The 2↔4 label swap made things worse.")
            print(f"   Consider trying different label swaps")
    
    print(f"\n📸 Check the verification_*.png files to see visual comparisons")

if __name__ == "__main__":
    main()