# ResDex

Official code for **"Efficient Residual Learning with Mixture-of-Experts for Universal Dexterous Grasping"** *(ICLR 2025)*

![Demo](fig/demo.gif)
*For detailed information, please refer to our [paper](https://openreview.net/pdf?id=BUj9VSCoET).*


## Installation

### 1. Clone repository:
```bash
git clone https://github.com/analyst-huang/ResDex.git
cd ResDex
```

### 2. Create conda environment:
```bash
conda create -n resdex python=3.8
conda activate resdex
```

### 3. Install IsaacGym
Download Isaac Gym Preview 4 [here](https://developer.nvidia.com/isaac-gym-preview-4) and following the installation document.

### 4. Install dependencies:
```bash
pip install -r requirements.txt
```

### 5. Install PointNet2:
```bash
cd third_party/pointnet2
python setup.py install
```

## Data Preparation
Download dataset from [here](INSERT_DATA_LINK_HERE) and organize as:
```
assets/
├── objects/
├── grasps/
└── annotations/
```

## Training & Evaluation

**Base Policy:**
```bash
bash scripts/train_blind.sh
```

**Residual Policy:**
```bash
bash scripts/train_residual.sh
```

**Evaluation:**
```bash
# Base policy
bash scripts/test_blind.sh

# Residual policy
bash scripts/test_residual.sh
```

**DAGGER Distillation:**
```bash
bash scripts/train_dagger_vision.sh
```

**Vision Policy Evaluation:**
```bash
bash scripts/test_dagger_vision.sh
```

## Acknowledgement
This project is built upon [UnidexGrasp](https://github.com/PKU-EPIC/UniDexGrasp).
