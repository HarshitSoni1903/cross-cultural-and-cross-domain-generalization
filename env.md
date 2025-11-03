# NYU HPC Multilang Environment Setup Guide
### This document provides a step-by-step reference for setting up and launching the Multilang environment on NYU HPC using Singularity with a writable overlay and Conda environment. It includes only the verified commands that worked correctly.
# 1. Base Environment Preparation
Create a working directory and copy the base overlay and Singularity image:

```bash
mkdir -p /scratch/$USER/multilang
cd /scratch/$USER/multilang
cp -rp /scratch/work/public/overlay-fs-ext3/overlay-50GB-10M.ext3.gz .
gunzip overlay-50GB-10M.ext3.gz
```

# 2. Multilang Launcher Creation
Paste this entire block into your terminal to create the launcher script that automatically starts Singularity with RW overlay and activates the Conda environment.

```bash
cat > ~/multilang <<'EOF'
#!/bin/bash

SCRATCH_DIR=/scratch/$USER/multilang
OVERLAY=${SCRATCH_DIR}/overlay-50G-10M.ext3
SIF=/scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
ENV_NAME=multilang

# Verify resources
if [ ! -f "$OVERLAY" ]; then
    echo "Error: Overlay not found at $OVERLAY" >&2
    exit 1
fi
if [ ! -f "$SIF" ]; then
    echo "Error: Singularity image not found at $SIF" >&2
    exit 1
fi
if [ ! -d "$SCRATCH_DIR" ]; then
    echo "Error: Scratch directory $SCRATCH_DIR not found." >&2
    exit 1
fi

# Change to scratch before launching
cd $SCRATCH_DIR || exit 1

# Start container with /scratch bound and overlay mounted
singularity exec --nv --bind /scratch --overlay ${OVERLAY}:rw ${SIF} /bin/bash -c "
    source /ext3/env.sh 2>/dev/null || {
        echo '/ext3/env.sh missing or invalid. Please recreate it.'
        exit 1
    }
    if ! conda env list | grep -q '^${ENV_NAME} '; then
        echo 'Conda environment ${ENV_NAME} not found. Creating now...'
        conda create -y -n ${ENV_NAME} python=3.11
    fi
    conda activate ${ENV_NAME}
    cd ${SCRATCH_DIR}
    exec /bin/bash
"
EOF

chmod +x ~/multilang
echo "alias multilang='~/multilang'" >> ~/.bashrc
source ~/.bashrc
```

# 3. One-Time Conda and Miniforge Setup
Run these commands manually inside the Singularity shell (launched manually once) to install Conda and prepare the base environment.

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3
source /ext3/miniforge3/etc/profile.d/conda.sh
conda create -y -n multilang python=3.11
conda activate multilang
conda install -y -c conda-forge numpy scipy pandas scikit-learn tqdm requests pyyaml
```

# 4. Create /ext3/env.sh
Create the environment initialization script inside Singularity to make Conda available automatically:

```bash
cat > /ext3/env.sh <<'EOF'
#!/bin/bash
export PATH=/ext3/miniforge3/bin:$PATH
source /ext3/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true
EOF
chmod +x /ext3/env.sh
```