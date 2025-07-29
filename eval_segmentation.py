#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
import wandb
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import glob

def load_nifti(path):
    """Load and return nifti data"""
    nii = nib.load(str(path))
    return nii.get_fdata()

def compute_segmentation_metrics(pred_seg, gt_seg):
    """Compute segmentation metrics"""
    # Convert to tensor format for MONAI metrics
    pred_tensor = torch.from_numpy(pred_seg).unsqueeze(0).unsqueeze(0)
    gt_tensor = torch.from_numpy(gt_seg).unsqueeze(0).unsqueeze(0)
    
    # Convert to one-hot for multi-class metrics
    pred_onehot = torch.zeros(1, 4, *pred_seg.shape)  # 4 classes (0,1,2,3)
    gt_onehot = torch.zeros(1, 4, *gt_seg.shape)
    
    for i in range(4):
        pred_onehot[0, i] = (pred_tensor[0, 0] == i).float()
        gt_onehot[0, i] = (gt_tensor[0, 0] == i).float()
        
    # Compute metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_scores = dice_metric(pred_onehot, gt_onehot)
    
    # Individual class dice scores
    dice_et = float(dice_scores[0, 2]) if len(dice_scores[0]) > 2 else 0.0  # Enhancing tumor  
    dice_tc = float(dice_scores[0, 1]) if len(dice_scores[0]) > 1 else 0.0  # Tumor core
    dice_wt = float(dice_scores[0, 3]) if len(dice_scores[0]) > 3 else 0.0  # Whole tumor
    
    return {
        "dice_et": dice_et,
        "dice_tc": dice_tc, 
        "dice_wt": dice_wt,
        "dice_mean": float(np.mean([dice_et, dice_tc, dice_wt]))
    }

def create_segmentation_overlay(image, pred_seg, gt_seg, case_id):
    """Create segmentation comparison visualization"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    slice_idx = image.shape[2] // 2
    img_slice = image[:, :, slice_idx]
    pred_slice = pred_seg[:, :, slice_idx]
    gt_slice = gt_seg[:, :, slice_idx]
    
    # Original image
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Predicted segmentation
    axes[1].imshow(pred_slice, cmap='jet', vmin=0, vmax=3)
    axes[1].set_title('Predicted Seg')
    axes[1].axis('off')
    
    # Ground truth segmentation
    axes[2].imshow(gt_slice, cmap='jet', vmin=0, vmax=3)
    axes[2].set_title('Ground Truth Seg')
    axes[2].axis('off')
    
    # Overlay comparison
    axes[3].imshow(img_slice, cmap='gray', alpha=0.7)
    axes[3].imshow(pred_slice, cmap='Reds', alpha=0.3, vmin=0, vmax=3)
    axes[3].imshow(gt_slice, cmap='Blues', alpha=0.3, vmin=0, vmax=3)
    axes[3].set_title('Overlay (Red=Pred, Blue=GT)')
    axes[3].axis('off')
    
    plt.suptitle(f'Segmentation Comparison - {case_id}')
    plt.tight_layout()
    return fig

def main():
    # Initialize wandb (continue the existing run)
    wandb.init(project="fast-cwmd-eval", entity="timgsereda", job_type="seg_evaluation")

    # Find prediction files
    pred_files = glob.glob("./outputs/*.nii.gz")
    print(f"Found {len(pred_files)} prediction files")

    # Original training data directory
    original_dir = Path("BraTS2023-TrainingData-Original")

    all_dice_scores = {"dice_et": [], "dice_tc": [], "dice_wt": [], "dice_mean": []}

    for pred_file in pred_files[:10]:  # Limit for testing
        case_name = Path(pred_file).stem
        print(f"Evaluating {case_name}")
        
        # Load predicted segmentation
        pred_seg = load_nifti(pred_file)
        
        # Find corresponding ground truth segmentation
        gt_seg_file = original_dir / f"{case_name}_seg.nii.gz"
        if not gt_seg_file.exists():
            print(f"Warning: GT segmentation not found for {case_name}")
            continue
            
        gt_seg = load_nifti(gt_seg_file)
        
        # Load one of the original modalities for visualization
        t1_file = original_dir / f"{case_name}_t1.nii.gz"
        if t1_file.exists():
            t1_img = load_nifti(t1_file)
        else:
            t1_img = None
        
        # Compute metrics
        metrics = compute_segmentation_metrics(pred_seg, gt_seg)
        
        # Add to aggregated results
        for key, value in metrics.items():
            if key in all_dice_scores:
                all_dice_scores[key].append(value)
        
        # Create visualization
        if t1_img is not None:
            fig = create_segmentation_overlay(t1_img, pred_seg, gt_seg, case_name)
            
            # Log to wandb
            wandb.log({
                f"seg_eval/{case_name}/comparison": wandb.Image(fig),
                f"seg_eval/{case_name}/dice_et": metrics["dice_et"],
                f"seg_eval/{case_name}/dice_tc": metrics["dice_tc"],
                f"seg_eval/{case_name}/dice_wt": metrics["dice_wt"],
                f"seg_eval/{case_name}/dice_mean": metrics["dice_mean"]
            })
            
            plt.close(fig)
        
        print(f"  Dice ET: {metrics['dice_et']:.3f}, TC: {metrics['dice_tc']:.3f}, WT: {metrics['dice_wt']:.3f}")

    # Compute and log summary statistics
    summary_stats = {}
    for metric, values in all_dice_scores.items():
        if values:
            summary_stats[f"seg_eval/summary/{metric}_mean"] = np.mean(values)
            summary_stats[f"seg_eval/summary/{metric}_std"] = np.std(values)

    wandb.log(summary_stats)

    print("=== SEGMENTATION EVALUATION SUMMARY ===")
    for metric, values in all_dice_scores.items():
        if values:
            print(f"{metric}: {np.mean(values):.3f} ± {np.std(values):.3f}")

    wandb.finish()

if __name__ == "__main__":
    main()