#!/bin/bash

# Fix nnUNet Setup for BraTS Segmentation
# Run this script to resolve the environment and dataset issues

echo "🔧 Fixing nnUNet setup..."

# 1. Set up environment variables (add to ~/.bashrc for persistence)
export nnUNet_raw="/app/nnunet/raw"
export nnUNet_preprocessed="/app/nnunet/preprocessed" 
export nnUNet_results="/app/nnunet/results"

# Make sure directories exist
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

echo "✅ Environment variables set:"
echo "   nnUNet_raw=$nnUNet_raw"
echo "   nnUNet_preprocessed=$nnUNet_preprocessed" 
echo "   nnUNet_results=$nnUNet_results"

# 2. Verify the pre-trained model structure exists
MODEL_DIR="$nnUNet_results/Dataset137_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_5"
echo "🔍 Checking model directory: $MODEL_DIR"

if [ ! -f "$MODEL_DIR/checkpoint_final.pth" ]; then
    echo "❌ Model checkpoint not found. Please ensure the pre-trained weights are downloaded."
    echo "Expected location: $MODEL_DIR/checkpoint_final.pth"
    exit 1
fi

if [ ! -f "$nnUNet_results/Dataset137_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json" ]; then
    echo "❌ dataset.json not found."
    echo "Expected location: $nnUNet_results/Dataset137_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json"
    exit 1
fi

echo "✅ Pre-trained model structure verified"

# 3. Set up the dataset link properly (alternative to symlink if permissions fail)
DATASET_SOURCE="./Dataset137_BraTS2021_inference"
DATASET_TARGET="$nnUNet_raw/Dataset137_BraTS2021"

if [ -d "$DATASET_SOURCE" ]; then
    echo "🔗 Setting up dataset link..."
    
    # Remove existing link/directory if it exists
    if [ -L "$DATASET_TARGET" ] || [ -d "$DATASET_TARGET" ]; then
        rm -rf "$DATASET_TARGET"
    fi
    
    # Try symbolic link first
    if ln -s "$(realpath $DATASET_SOURCE)" "$DATASET_TARGET" 2>/dev/null; then
        echo "✅ Symbolic link created successfully"
    else
        echo "⚠️  Symbolic link failed, copying dataset instead..."
        cp -r "$DATASET_SOURCE" "$DATASET_TARGET"
        echo "✅ Dataset copied successfully"
    fi
else
    echo "❌ Dataset source not found: $DATASET_SOURCE"
    echo "Please run the conversion script first (conv.py or Dataset137_BraTS21.py)"
    exit 1
fi

# 4. Verify dataset structure
echo "🔍 Verifying dataset structure..."
if [ -d "$DATASET_TARGET/imagesTs" ]; then
    NUM_CASES=$(find "$DATASET_TARGET/imagesTs" -name "*_0000.nii.gz" | wc -l)
    echo "✅ Found $NUM_CASES cases in imagesTs directory"
    
    # Show first few files for verification
    echo "📁 Sample files:"
    ls "$DATASET_TARGET/imagesTs" | head -8
else
    echo "❌ imagesTs directory not found in $DATASET_TARGET"
    exit 1
fi

# 5. Test nnUNet can find the dataset
echo "🧪 Testing nnUNet dataset recognition..."
if python -c "
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
try:
    name = convert_id_to_dataset_name(137)
    print(f'✅ Dataset 137 found: {name}')
except Exception as e:
    print(f'❌ Dataset 137 not recognized: {e}')
    exit(1)
"; then
    echo "✅ nnUNet can recognize dataset 137"
else
    echo "❌ nnUNet cannot recognize dataset 137"
    exit 1
fi

echo ""
echo "🎉 Setup complete! You can now run:"
echo "   nnUNetv2_predict -i \"$DATASET_TARGET/imagesTs\" -o \"./outputs\" -d 137 -c 3d_fullres -f 5"
echo ""
echo "💡 To make environment variables persistent, add these lines to ~/.bashrc:"
echo "   export nnUNet_raw=\"$nnUNet_raw\""
echo "   export nnUNet_preprocessed=\"$nnUNet_preprocessed\""
echo "   export nnUNet_results=\"$nnUNet_results\""