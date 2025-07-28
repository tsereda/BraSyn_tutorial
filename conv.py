import multiprocessing
import shutil
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def convert_labels_back_to_BraTS(seg: np.ndarray):
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2
    new_seg[seg == 3] = 4
    new_seg[seg == 2] = 1
    return new_seg


def load_convert_labels_back_to_BraTS(filename, input_folder, output_folder):
    a = sitk.ReadImage(join(input_folder, filename))
    b = sitk.GetArrayFromImage(a)
    c = convert_labels_back_to_BraTS(b)
    d = sitk.GetImageFromArray(c)
    d.CopyInformation(a)
    sitk.WriteImage(d, join(output_folder, filename))


def convert_folder_with_preds_back_to_BraTS_labeling_convention(input_folder: str, output_folder: str, num_processes: int = 12):
    """
    reads all prediction files (nifti) in the input folder, converts the labels back to BraTS convention and saves the
    """
    maybe_mkdir_p(output_folder)
    nii = subfiles(input_folder, suffix='.nii.gz', join=False)
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        p.starmap(load_convert_labels_back_to_BraTS, zip(nii, [input_folder] * len(nii), [output_folder] * len(nii)))


if __name__ == '__main__':
    # ✅ FIXED: Change to your actual completed validation directory
    brats_data_dir = 'completed_validation'  # or wherever your synthesis outputs go
    
    task_id = 137
    task_name = "BraTS2021_inference"  # Changed name to indicate inference

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders (images only - no labels for inference)
    out_base = join('./', foldername)
    imagestr = join(out_base, "imagesTs")  # ✅ FIXED: Use imagesTs for test/inference
    maybe_mkdir_p(imagestr)
    # ❌ REMOVED: No labelstr needed for inference

    # Get case directories
    case_ids = subdirs(brats_data_dir, prefix='BraTS', join=False)
    
    print(f"Found {len(case_ids)} cases for inference")
    print(f"First few cases: {case_ids[:3]}")

    for c in case_ids:
        print(f"Processing case: {c}")
        
        # Check if all 4 modalities exist
        t1n_file = join(brats_data_dir, c, c + "-t1n.nii.gz")
        t1c_file = join(brats_data_dir, c, c + "-t1c.nii.gz") 
        t2w_file = join(brats_data_dir, c, c + "-t2w.nii.gz")
        t2f_file = join(brats_data_dir, c, c + "-t2f.nii.gz")
        
        missing_files = []
        if not exists(t1n_file): missing_files.append("t1n")
        if not exists(t1c_file): missing_files.append("t1c") 
        if not exists(t2w_file): missing_files.append("t2w")
        if not exists(t2f_file): missing_files.append("t2f")
        
        if missing_files:
            print(f"  ❌ Skipping {c} - missing: {missing_files}")
            continue
            
        # Copy files in nnUNet format
        shutil.copy(t1n_file, join(imagestr, c + '_0000.nii.gz'))
        shutil.copy(t1c_file, join(imagestr, c + '_0001.nii.gz'))
        shutil.copy(t2w_file, join(imagestr, c + '_0002.nii.gz'))
        shutil.copy(t2f_file, join(imagestr, c + '_0003.nii.gz'))
        
        print(f"  ✅ Converted {c}")

    # ❌ REMOVED: No dataset.json generation needed for inference
    # The pre-trained model already has its dataset.json
    
    print(f"\n✅ Conversion complete!")
    print(f"📁 nnUNet inference data ready at: {imagestr}")
    print(f"📝 Found {len([f for f in os.listdir(imagestr) if f.endswith('_0000.nii.gz')])} cases")
    
    # Create symlink for nnUNet (if needed)
    nnunet_dataset_path = join(nnUNet_raw, foldername)
    if not exists(nnunet_dataset_path):
        os.symlink(os.path.abspath(out_base), nnunet_dataset_path)
        print(f"🔗 Created symlink: {nnunet_dataset_path}")