import multiprocessing
import shutil
import os
from multiprocessing import Pool
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_raw

def convert_labels_back_to_BraTS(seg: np.ndarray):
    """Convert nnUNet labels back to BraTS convention"""
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2
    new_seg[seg == 3] = 4
    new_seg[seg == 2] = 1
    return new_seg

def load_convert_labels_back_to_BraTS(filename, input_folder, output_folder):
    """Load segmentation, convert labels, and save"""
    a = sitk.ReadImage(join(input_folder, filename))
    b = sitk.GetArrayFromImage(a)
    c = convert_labels_back_to_BraTS(b)
    d = sitk.GetImageFromArray(c)
    d.CopyInformation(a)
    sitk.WriteImage(d, join(output_folder, filename))

def convert_folder_with_preds_back_to_BraTS_labeling_convention(input_folder: str, output_folder: str, num_processes: int = 12):
    """Convert all prediction files back to BraTS labeling convention"""
    maybe_mkdir_p(output_folder)
    nii = subfiles(input_folder, suffix='.nii.gz', join=False)
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        p.starmap(load_convert_labels_back_to_BraTS, zip(nii, [input_folder] * len(nii), [output_folder] * len(nii)))

if __name__ == '__main__':
    # 🔧 CORRECTED: Updated path to match the synthesis output directory
    # Adjust this path based on where your synthesis actually outputs the completed data
    
    # Option 1: If synthesis outputs to pseudo_val_set (from your original script)
    brats_data_dir = './pseudo_val_set'
    
    # Option 2: If synthesis outputs to different location (uncomment and modify as needed)
    # brats_data_dir = '../brats-synthesis/datasets/BRATS2023/pseudo_validation_completed'
    
    # 🔧 IMPROVEMENT: Check if directory exists
    if not os.path.exists(brats_data_dir):
        print(f"❌ Data directory not found: {brats_data_dir}")
        print("Please check the path to your completed synthesis data.")
        print("Common locations might be:")
        print("  - ./pseudo_val_set")
        print("  - ../brats-synthesis/datasets/BRATS2023/pseudo_validation_completed")
        print("  - ../brats-synthesis/outputs")
        exit(1)
    
    task_id = 137
    task_name = "BraTS2021_inference"
    foldername = "Dataset%03.0d_%s" % (task_id, task_name)
    
    # Setting up nnU-Net folders for inference
    out_base = join('./', foldername)
    imagestr = join(out_base, "imagesTs")  # Use imagesTs for test/inference data
    maybe_mkdir_p(imagestr)
    
    print(f"🔍 Looking for data in: {brats_data_dir}")
    
    # Get case directories - handle different possible structures
    if os.path.exists(brats_data_dir):
        # Try to find BraTS case directories
        potential_cases = [d for d in os.listdir(brats_data_dir) 
                          if os.path.isdir(join(brats_data_dir, d)) and 'BraTS' in d]
        
        if not potential_cases:
            # Maybe files are directly in the directory
            potential_cases = [d for d in os.listdir(brats_data_dir) 
                              if os.path.isdir(join(brats_data_dir, d))]
        
        case_ids = potential_cases
    else:
        case_ids = []
    
    if not case_ids:
        print(f"❌ No case directories found in {brats_data_dir}")
        print("Directory contents:")
        try:
            for item in os.listdir(brats_data_dir):
                print(f"  - {item}")
        except:
            print("  (directory not accessible)")
        exit(1)
    
    print(f"Found {len(case_ids)} cases for inference")
    print(f"First few cases: {case_ids[:3]}")
    
    processed_count = 0
    
    for c in case_ids:
        print(f"Processing case: {c}")
        
        case_dir = join(brats_data_dir, c)
        
        # 🔧 IMPROVEMENT: More flexible file finding
        # Look for files with different possible naming patterns
        possible_patterns = [
            (c + "-t1n.nii.gz", c + "-t1c.nii.gz", c + "-t2w.nii.gz", c + "-t2f.nii.gz"),
            (c + "_t1n.nii.gz", c + "_t1c.nii.gz", c + "_t2w.nii.gz", c + "_t2f.nii.gz"),
            ("t1n.nii.gz", "t1c.nii.gz", "t2w.nii.gz", "t2f.nii.gz"),
        ]
        
        files_found = None
        for pattern in possible_patterns:
            t1n_file = join(case_dir, pattern[0])
            t1c_file = join(case_dir, pattern[1])
            t2w_file = join(case_dir, pattern[2])
            t2f_file = join(case_dir, pattern[3])
            
            if all(os.path.exists(f) for f in [t1n_file, t1c_file, t2w_file, t2f_file]):
                files_found = (t1n_file, t1c_file, t2w_file, t2f_file)
                break
        
        if files_found is None:
            print(f"  ❌ Skipping {c} - could not find all 4 modalities")
            print(f"     Looked in: {case_dir}")
            print(f"     Available files: {os.listdir(case_dir) if os.path.exists(case_dir) else 'directory not found'}")
            continue
        
        # Copy files in nnUNet format (channel order: T1n, T1c, T2w, T2f)
        try:
            shutil.copy(files_found[0], join(imagestr, c + '_0000.nii.gz'))  # T1n
            shutil.copy(files_found[1], join(imagestr, c + '_0001.nii.gz'))  # T1c
            shutil.copy(files_found[2], join(imagestr, c + '_0002.nii.gz'))  # T2w
            shutil.copy(files_found[3], join(imagestr, c + '_0003.nii.gz'))  # T2f
            
            processed_count += 1
            print(f"  ✅ Converted {c}")
            
        except Exception as e:
            print(f"  ❌ Error copying files for {c}: {e}")
            continue
    
    print(f"\n✅ Conversion complete!")
    print(f"📁 nnUNet inference data ready at: {imagestr}")
    print(f"📝 Successfully processed {processed_count} cases")
    
    if processed_count == 0:
        print("❌ No cases were successfully processed!")
        print("Please check your data directory structure and file naming.")
        exit(1)
    
    # 🔧 IMPROVEMENT: Better error handling for symlink creation
    try:
        if nnUNet_raw is not None:
            nnunet_dataset_path = join(nnUNet_raw, foldername)
            
            # Remove existing link/directory if it exists
            if os.path.exists(nnunet_dataset_path) or os.path.islink(nnunet_dataset_path):
                if os.path.islink(nnunet_dataset_path):
                    os.unlink(nnunet_dataset_path)
                else:
                    shutil.rmtree(nnunet_dataset_path)
            
            # Create new symlink
            os.symlink(os.path.abspath(out_base), nnunet_dataset_path)
            print(f"🔗 Created symlink: {nnunet_dataset_path}")
        else:
            print("⚠️  nnUNet_raw not set - skipping symlink creation")
            
    except Exception as e:
        print(f"⚠️  Could not create nnUNet symlink: {e}")
        print(f"   You can manually copy the dataset to your nnUNet_raw folder:")
        print(f"   cp -r {os.path.abspath(out_base)} $nnUNet_raw/{foldername}")
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Make sure nnUNet environment variables are set")
    print(f"   2. Run: nnUNetv2_predict -i \"{imagestr}\" -o \"./outputs\" -d 137 -c 3d_fullres -f 5")