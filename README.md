# Hematology AI: Anemia Risk AI & Precision Nutrition Pipeline

An interactive, multi-omic diagnostic and precision nutrition platform. The application combines clinical phenotype features (NHANES-modeled) and genomics (GWAS-modeled Polygenic Risk Scores) to predict anemia susceptibility and generate customized nutrition plans based on genetic pathway burden.

## 🚀 Key Features

* **Consolidated Precision Advisor**: A unified dashboard matching patient phenotypes, VCF files, eating habits, and food allergies to calculate risk scores and formulate targeted diets in a single execution.
* **Genomic VCF Integration**: A drag-and-drop genomic parser that scans patient VCF sequences for key SNP variants (e.g., *TMPRSS6*, *ACSL3P1*, *PIEZO1*, *FADS1*), calculates Cohort Polygenic Risk Scores (PRS), and highlights metabolic pathway dysfunction.
* **XGBoost Machine Learning Engine**: 
  * **Classifier**: Predicts risk category (Low, Moderate, High Susceptibility) based on balanced SMOTE training.
  * **Regressor**: Forecasts exact predicted Hemoglobin (Hb) levels.
* **Training Center**: Real-time diagnostic panel to monitor training status, check validation metrics (AUC, MAE, R², F1), and evaluate class balance distributions.
* **Whole-Food Dietary Framework**: Automatically maps diet options, flags clinical warnings, and suggests synergistic food pairings (e.g., Vitamin C + Iron absorption combinations).

---

## 🛠️ Architecture & Tech Stack

* **Backend**: FastAPI (Python), XGBoost, Scikit-Learn, Pandas, Numpy, Uvicorn
* **Frontend**: HTML5, Vanilla JS (ES6+), CSS3 variables, FontAwesome Icons
* **Data Sources**: NHANES phenotype database (synthetic simulation) + GWAS genomic risk models

---

## 💻 Installation & Setup

### 1. Prerequisites
Make sure you have Python 3.9+ installed on your system.

### 2. Clone and Initialize Virtual Environment
```bash
# Navigate to workspace
cd idpfood

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Development Server
```bash
python main.py
```
The server will start at `http://127.0.0.1:8000/`.

---

## 📂 Project Structure

```
idpfood/
├── main.py                # FastAPI Application & XGBoost Training Pipeline
├── requirements.txt       # Project Dependencies
├── README.md              # Documentation
└── static/                # Frontend Assets
    ├── index.html         # Main Dashboard Layout
    ├── style.css          # Modern Dark Glassmorphism Styling
    ├── app.js             # API Integration & Gauge Animations
    └── sample_patient.vcf # Mock VCF for Testing Genomic Uploads
```

---

## 🧪 Quick Test Guide

1. Open `http://127.0.0.1:8000/` in your browser.
2. Go to the **Precision Advisor** tab.
3. Click one of the **Clinical Case Templates** (e.g., *Severe Anemia - Hepcidin Axis*) to pre-populate clinical inputs.
4. Drag and drop the `sample_patient.vcf` file from `static/` into the upload box (or manually configure the mock SNPs).
5. Click **Calculate Anemia Risk & Generate Diet**.
6. View the unified Multi-Omic Diagnostic Report, Precision Nutrition Plan, and Whole-Food Dietary Framework.
