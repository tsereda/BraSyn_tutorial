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

def compute_dice_score(pred_mask, gt_mask):
    """Compute Dice score between two binary masks"""
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    
    if union > 0:
        return (2.0 * intersection) / union
    else:
        return 1.0 if np.sum(pred_mask) == 0 else 0.0

def analyze_case(pred_seg, gt_seg, case_name):
    """Analyze one case and return results"""
    print(f"\n🔍 ANALYZING: {case_name}")
    print("=" * 50)
    
    # Compute Dice for each label
    results = {}
    label_names = {
        1: "Necrotic/Non-enhancing (NCR/NET)",
        2: "Peritumoral Edema (ED)", 
        3: "Enhancing Tumor (ET)",
        4: "Whole Tumor (WT)"
    }
    
    print(f"{'Label':<8} {'Name':<30} {'Dice':<8} {'Status'}")
    print("-" * 60)
    
    dice_scores = []
    for label in [1, 2, 3, 4]:
        pred_mask = (pred_seg == label).astype(float)
        gt_mask = (gt_seg == label).astype(float)
        
        dice = compute_dice_score(pred_mask, gt_mask)
        results[label] = dice
        dice_scores.append(dice)
        
        # Status based on performance
        if dice > 0.8:
            status = "🟢 Excellent"
        elif dice > 0.5:
            status = "🟡 Good" 
        elif dice > 0.1:
            status = "🟠 Poor"
        else:
            status = "🔴 Very Poor"
            
        print(f"{label:<8} {label_names[label]:<30} {dice:.3f}    {status}")
    
    avg_dice = np.mean(dice_scores)
    print("-" * 60)
    print(f"{'AVERAGE':<39} {avg_dice:.3f}")
    
    return results, avg_dice

def create_detailed_comparison(pred_seg, gt_seg, case_name, slice_idx=None):
    """Create detailed visual comparison"""
    if slice_idx is None:
        slice_idx = pred_seg.shape[2] // 2
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    pred_slice = pred_seg[:, :, slice_idx]
    gt_slice = gt_seg[:, :, slice_idx]
    
    # Top row: Full segmentations
    axes[0, 0].imshow(pred_slice, cmap='jet', vmin=0, vmax=4)
    axes[0, 0].set_title('Prediction - Full')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_slice, cmap='jet', vmin=0, vmax=4)
    axes[0, 1].set_title('Ground Truth - Full')
    axes[0, 1].axis('off')
    
    # Overlay comparison
    axes[0, 2].imshow(pred_slice, cmap='Reds', alpha=0.7, vmin=0, vmax=4)
    axes[0, 2].imshow(gt_slice, cmap='Blues', alpha=0.3, vmin=0, vmax=4)
    axes[0, 2].set_title('Overlay (Red=Pred, Blue=GT)')
    axes[0, 2].axis('off')
    
    # Difference map
    diff = np.abs(pred_slice - gt_slice)
    axes[0, 3].imshow(diff, cmap='hot')
    axes[0, 3].set_title('Difference Map')
    axes[0, 3].axis('off')
    
    # Bottom row: Individual labels
    colors = ['Purples', 'Blues', 'Greens', 'Oranges']
    labels = [1, 2, 3, 4]
    
    for i, (label, cmap) in enumerate(zip(labels, colors)):
        pred_label = (pred_slice == label).astype(float)
        gt_label = (gt_slice == label).astype(float)
        
        # Combine prediction (red) and GT (blue)
        combined = np.zeros((*pred_slice.shape, 3))
        combined[:, :, 0] = pred_label  # Red channel for prediction
        combined[:, :, 2] = gt_label    # Blue channel for GT
        # Purple where they overlap
        
        axes[1, i].imshow(combined)
        axes[1, i].set_title(f'Label {label}\n(Red=Pred, Blue=GT, Purple=Both)')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Detailed Analysis: {case_name} (Slice {slice_idx})')
    plt.tight_layout()
    
    # Save the plot
    plot_file = f"analysis_{case_name}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📸 Analysis plot saved as: {plot_file}")
    
    return fig

def suggest_swaps(all_results):
    """Analyze all cases and suggest potential label swaps"""
    print(f"\n" + "="*70)
    print("🎯 SWAP ANALYSIS ACROSS ALL CASES")
    print("="*70)
    
    # Average performance per label
    label_performance = {1: [], 2: [], 3: [], 4: []}
    
    for case_name, results in all_results.items():
        for label, dice in results.items():
            label_performance[label].append(dice)
    
    avg_performance = {}
    for label, scores in label_performance.items():
        avg_performance[label] = np.mean(scores) if scores else 0.0
    
    print("Average Dice per label:")
    for label, avg_dice in avg_performance.items():
        print(f"  Label {label}: {avg_dice:.3f}")
    
    # Find problematic labels
    poor_labels = [label for label, dice in avg_performance.items() if dice < 0.3]
    good_labels = [label for label, dice in avg_performance.items() if dice > 0.7]
    
    print(f"\n🔴 Poor performing labels (< 0.3): {poor_labels}")
    print(f"🟢 Good performing labels (> 0.7): {good_labels}")
    
    if poor_labels and good_labels:
        print(f"\n💡 SUGGESTIONS:")
        print(f"   - Your model performs well overall!")
        print(f"   - Labels {good_labels} are working excellently")
        if poor_labels:
            print(f"   - Labels {poor_labels} might need attention, but check if they exist in GT")
    else:
        print(f"\n✅ CONCLUSION: Your segmentation looks good! No swaps needed.")
        print(f"   Average performance > 0.7 suggests the model is working correctly.")

def main():
    print("🧠 BraTS Segmentation Analysis Tool")
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
    
    print(f"✅ Found training data directory")
    
    all_results = {}
    all_dice_scores = []
    
    # Process each case
    for pred_file in pred_files:
        case_name = Path(pred_file).name.replace('.nii.gz', '').replace('.nii', '')
        
        # Find ground truth
        case_dir = training_data_dir / case_name
        if not case_dir.exists():
            print(f"⚠️  Case directory not found: {case_dir}")
            continue
            
        gt_file = case_dir / f"{case_name}-seg.nii.gz"
        if not gt_file.exists():
            print(f"⚠️  Ground truth not found: {gt_file}")
            continue
        
        # Load files
        try:
            pred_seg = load_nifti(pred_file)
            gt_seg = load_nifti(gt_file)
        except Exception as e:
            print(f"❌ Error loading {case_name}: {e}")
            continue
        
        # Analyze this case
        results, avg_dice = analyze_case(pred_seg, gt_seg, case_name)
        all_results[case_name] = results
        all_dice_scores.append(avg_dice)
        
        # Create visualization
        create_detailed_comparison(pred_seg, gt_seg, case_name)
    
    # Overall summary
    if all_dice_scores:
        overall_avg = np.mean(all_dice_scores)
        print(f"\n" + "="*70)
        print(f"📊 OVERALL PERFORMANCE SUMMARY")
        print("="*70)
        print(f"Overall Average Dice: {overall_avg:.3f}")
        print(f"Cases analyzed: {len(all_dice_scores)}")
        
        if overall_avg > 0.7:
            print("🎉 EXCELLENT! Your model is performing very well!")
        elif overall_avg > 0.5:
            print("👍 GOOD! Solid performance with room for improvement")
        elif overall_avg > 0.3:
            print("⚠️  MODERATE: Performance could be improved")
        else:
            print("❌ POOR: Significant issues need attention")
        
        # Suggest potential swaps
        suggest_swaps(all_results)
    
    print(f"\n📸 Check analysis_*.png files for detailed visualizations")
    print(f"🎯 Based on the analysis above, decide if any label swaps are needed")

if __name__ == "__main__":
    main()