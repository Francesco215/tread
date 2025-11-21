#!/bin/bash
set -e

echo "Setting up FID prerequisites..."

# Create directories
mkdir -p assets/fid_stats

# Download Inception V3 model
if [ ! -f assets/inception-2015-12-05.pkl ]; then
    echo "Downloading Inception V3 model..."
    wget -O assets/inception-2015-12-05.pkl https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl
else
    echo "assets/inception-2015-12-05.pkl already exists."
fi

# Download Reference Statistics
if [ ! -f assets/fid_stats/VIRTUAL_imagenet256_labeled.npz ]; then
    echo "Downloading Reference Statistics..."
    wget -O assets/fid_stats/VIRTUAL_imagenet256_labeled.npz https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
else
    echo "assets/fid_stats/VIRTUAL_imagenet256_labeled.npz already exists."
fi

echo "FID setup complete."
