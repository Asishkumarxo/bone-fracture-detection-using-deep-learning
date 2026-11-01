# Reproducibility and Computational Environment Guide
**Project:** Bone Fracture Image Captioning & Detection Using Deep Learning  
**Scope:** Step-by-step workflow reproduction, dependency pinning, hardware configurations, checkpoint checksums, and dataset governance.  
**Source Code References:** [`requirements.txt`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/requirements.txt), [`requirements-dev.txt`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/requirements-dev.txt), [`.env.example`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/.env.example)  
**Document Version:** 1.0 (Audited October 2026)

---

## 1. Computational Environment Specifications

* **Operating System:** Tested and fully compatible with:
  - Windows 10 / Windows 11 (64-bit)
  - Ubuntu Linux 20.04 LTS / 22.04 LTS
  - macOS Monterey / Ventura / Sonoma (Apple Silicon MPS & Intel)
* **Python Runtime:** Python **3.10**, **3.11**, **3.12**, **3.13**, or **3.14** (verified on Python 3.14.7 in local `.venv`).
* **Hardware Profile:**
  - **CPU Execution:** Minimum 4 CPU cores, 8 GB System RAM (16 GB recommended for full 47,931-image graph clustering).
  - **GPU Acceleration (Optional for inference, recommended for training):**
    - NVIDIA GPU with CUDA 11.8 or 12.1+ compute capability.
    - Verified locally on an **NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB VRAM)**.
    - Peak VRAM footprint: 3,420 MB (ResNet-50 448×448, Batch 16, AMP); 685.8 MB (Inception-v3 299×299, Batch 8, AMP).
* **Random Seeds:** Default random seed $42$ configured across NumPy, Python standard library `random`, and PyTorch (`torch.manual_seed(42)`).

---

## 2. Pinned Dependencies Ledger

### 2.1 Core Runtime Dependencies (`requirements.txt`)
```ini
# Deep Learning & Tensor Framework
torch>=2.1.0
torchvision>=0.16.0

# Scientific Computing & Data Processing
numpy>=1.24.0,<3.0.0
pandas>=2.0.0
Pillow>=10.0.0
scikit-learn>=1.3.0

# Diagnostic Visualization & Plots
matplotlib>=3.7.0
seaborn>=0.12.0

# Production FastAPI REST Backend
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6
pydantic>=2.0.0

# Clinical Streamlit Web Frontend
requests>=2.31.0
streamlit>=1.30.0

# HTTP Testing Client
httpx>=0.25.0
```

### 2.2 Development & Quality Assurance Dependencies (`requirements-dev.txt`)
```ini
-r requirements.txt
pytest>=7.4.0
pytest-asyncio>=0.21.0
flake8>=6.1.0
black>=23.7.0
python-docx>=1.1.0
reportlab>=4.0.0
```

---

## 3. Checkpoint Provenance & Checksum Ledger

All evaluation metrics and inference services rely on five static model checkpoints stored under `models/checkpoints/`:

| Checkpoint Filename | Architecture | Role | File Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :--- | :---: | :--- |
| `best_model.pt` | ResNet-50 Multi-Task ($224\times 224$) | Deployed low-resolution classifier | 100,688,398 | `e685f5e27a696c738743d57e6274488be62283a0058b8f2d5930e12353a25cb4` |
| `best_detector.pt` | Faster R-CNN MobileNetV3 FPN ($384\times 384$) | Deployed lesion localization detector | 76,016,634 | `321b7a2d480e61d8583ea23cfbc89178ad5106cebceea0e1f72a4feaa639aa29` |
| `exp3_resnet50_448.pt` | ResNet-50 Multi-Task ($448\times 448$) | Deployed high-resolution classifier | 100,690,906 | `f31533c52d58ae756d0ed93691513e19f5f29442498a0882fe467d5d91986dfb` |
| `exp4_balanced_resnet50_448.pt` | Balanced ResNet-50 ($448\times 448$) | **Top clinical performer** (Sensitivity winner) | 100,695,553 | `6aaea982892246e16455e3525b58aabdcd4315edf1974b24864b7052a8486d83` |
| `exp5_inception_v3_multitask.pt`| Inception-v3 Multi-Task ($299\times 299$) | Specialized Foot & specificity benchmark | 105,270,015 | `3a22c513d4ba0a692b04cec63e57b65f9b2c2c05c43736213e847131f0f00b98` |

---

## 4. Step-by-Step Reproduction Workflow

### Step 1: Environment Setup
```powershell
# 1. Clone repository
git clone https://github.com/Asishkumarxo/bone-fracture-detection-using-deep-learning.git
cd bone-fracture-detection-using-deep-learning

# 2. Create virtual environment
python -m venv .venv

# 3. Activate environment
# On Windows PowerShell:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements-dev.txt

# 5. Configure environment file
Copy-Item .env.example .env
```

### Step 2: Dataset Preparation
1. Download `BoneFract A Bone Fracture Dataset.zip` from Mendeley Data (`doi:10.17632/jhqrxmff7v`).
2. Extract the archive into `data/BoneFract/`.
3. Download FracAtlas from Figshare (`doi:10.6084/m9.figshare.22634356`) into `data/FracAtlas/`.

### Step 3: Run Patient-Cluster Preprocessing & Split Generation
```powershell
python preprocess.py
```
This script executes:
- Bipartite graph clustering to isolate patient clusters.
- Stratified 80/10/10 partitioning into `train.csv`, `validation.csv`, and `test.csv`.
- Aspect-ratio preserved padding to standardized PNG formats in `processed/`.

### Step 4: Model Training Commands

#### A. Train Baseline ResNet-50 (224×224):
```powershell
python train.py --backbone resnet50 --epochs_stage1 3 --epochs_stage2 2 --batch_size 32
```

#### B. Train Experiment 3 (ResNet-50 at 448×448):
```powershell
python scripts/train_exp3_448.py
```

#### C. Train Experiment 4 (Balanced Region-Aware ResNet-50 at 448×448):
```powershell
python scripts/train_exp4_balanced.py
```

#### D. Train Experiment 5 (Multi-Task Inception-v3 at 299×299):
```powershell
python scripts/train_exp5_inception_v3.py
```

#### E. Train Faster R-CNN Lesion Detector:
```powershell
python train_detector.py --img_size 384 --epochs 5 --batch_size 4
```

### Step 5: Test Evaluation & Metrics Reproduction
To reproduce the benchmark metrics on the untouched test split ($N=4,778$):
```powershell
# Evaluate Experiment 3 (ResNet-50 448x448)
python scripts/evaluate_exp3_448.py

# Evaluate Experiment 4 (Balanced ResNet-50 448x448)
python scripts/evaluate_exp4.py

# Evaluate Experiment 5 (Multi-Task Inception-v3 299x299 with transform_input=True)
python scripts/evaluate_exp5.py
```

### Step 6: Automated Test Suite Verification
```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```
Expected output: **69 passed, 8 warnings in ~24s**.

### Step 7: Application Startup
```powershell
# Terminal 1: Launch FastAPI REST Backend
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Launch Streamlit Diagnostic Dashboard
streamlit run frontend/app.py
```

---

## 5. Reproducibility Blockers & Data Governance Notices

1. **Dataset Exclusions from Version Control:**  
   The raw dataset archives (`BoneFract A Bone Fracture Dataset.zip` ~3.95 GB and FracAtlas ~2.8 GB) are excluded from Git to comply with repository size guidelines and medical data governance. They must be downloaded from Mendeley Data and Figshare using the URLs in [`01_DATASETS_AND_DATA_PROVENANCE.md`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/docs/project_technical_documentation/01_DATASETS_AND_DATA_PROVENANCE.md).
2. **Deterministic Checkpoint Binaries:**  
   All evaluation metrics reported in this documentation were generated directly from the frozen checkpoint binaries stored under `models/checkpoints/`. Re-training with stochastic GPU operations (e.g., cuDNN nondeterministic algorithms) may cause minor variances in floating-point weights, but validation PR-AUC convergence trends remain identical.
