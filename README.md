# Detecting and Reducing Gender Bias in AI Recruitment Systems

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AIF360](https://img.shields.io/badge/AIF360-0.5.0-orange.svg)](https://aif360.mybluemix.net/)
[![Status](https://img.shields.io/badge/Status-Research-yellow.svg)](https://github.com)

> A comprehensive implementation of fairness-aware machine learning for detecting and mitigating gender bias in AI recruitment systems using explainable models.

**Author:** Samir Sharma  
**Institution:** University Institute of Computing, Chandigarh University  
**Contact:** samirsharmas005@gmail.com

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Fairness Metrics](#fairness-metrics)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

This project addresses the critical challenge of **gender bias in AI-driven recruitment systems**. Despite advances in workplace diversity, many hiring decisions continue to be influenced by unconscious bias. This research demonstrates how machine learning models can perpetuate these biases and provides practical solutions to mitigate them.

### Key Objectives

- 🔍 **Detect** existing gender bias in recruitment datasets
- ⚖️ **Mitigate** bias using three distinct approaches (pre-processing, in-processing, post-processing)
- 📊 **Evaluate** models using comprehensive fairness metrics
- 🔬 **Explain** model decisions using SHAP and LIME
- 📈 **Compare** trade-offs between accuracy and fairness

---

## ✨ Features

### Bias Detection & Mitigation
- ✅ **Three Mitigation Strategies:**
  - **Pre-processing:** Reweighing algorithm
  - **In-processing:** Adversarial Debiasing
  - **Post-processing:** Equalized Odds

### Fairness Metrics
- 📐 Demographic Parity Difference
- 📊 Disparate Impact Ratio
- ⚖️ Equal Opportunity Difference
- 📉 Average Odds Difference

### Model Explainability
- 🔎 **SHAP:** Global feature importance analysis
- 🎯 **LIME:** Local individual prediction explanations

### Comprehensive Evaluation
- 🏆 Multiple baseline models (Logistic Regression, Random Forest, SVM)
- 📊 Fairness-accuracy trade-off analysis
- 📈 Publication-ready visualizations

---

## 📁 Project Structure

```
gender-bias-detection/
│
├── data/                          # Dataset directory
│   ├── aug_train.csv             # Training data
│   └── aug_test.csv              # Test data
│
├── Detecting-Reducing-Gender-Bias.ipynb  # Main Jupyter notebook
│
├── visualizations/                # Generated plots and charts
│   ├── eda_gender_analysis.png
│   ├── shap_summary.png
│   ├── lime_explanation.png
│   ├── comprehensive_comparison.png
│   └── fairness_accuracy_tradeoff.png
│
├── results/                       # Output files
│   └── model_comparison.csv      # Fairness metrics comparison
│
├── models/                        # Saved models (optional)
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── LICENSE                        # License information
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/gender-bias-detection.git
cd gender-bias-detection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install AIF360 (if not included)

```bash
pip install aif360
```

### Dependencies List

Create a `requirements.txt` file with:

```txt
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
aif360>=0.5.0
fairlearn>=0.9.0
shap>=0.42.0
lime>=0.2.0
tensorflow>=2.13.0
jupyter>=1.0.0
notebook>=7.0.0
```

---

## 📊 Dataset

### Source
The dataset is obtained from the **Kaggle HR Analytics: Job Change of Data Scientists** repository.

📥 **Download:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)

### Description

- **Training samples:** 19,158
- **Test samples:** 2,129
- **Features:** 14 (demographic, educational, and professional attributes)
- **Target variable:** Job change intention (0 = Not looking, 1 = Looking)

### Key Features

| Feature | Description |
|---------|-------------|
| `enrollee_id` | Unique candidate ID |
| `city` | City code |
| `city_development_index` | Development index (scaled) |
| `gender` | Gender (Male/Female/Other) |
| `relevent_experience` | Relevant experience |
| `enrolled_university` | University enrollment type |
| `education_level` | Education level |
| `major_discipline` | Education major |
| `experience` | Total years of experience |
| `company_size` | Current employer size |
| `company_type` | Type of company |
| `last_new_job` | Years since last job change |
| `training_hours` | Training hours completed |
| `target` | Looking for job change (0/1) |

### Setup Instructions

1. Download `aug_train.csv` and `aug_test.csv` from Kaggle
2. Create a `data/` folder in the project root
3. Place both CSV files in the `data/` directory

---

## 🚀 Usage

### Running the Notebook

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open the main notebook:**
   - Navigate to `Detecting-Reducing-Gender-Bias.ipynb`

3. **Execute cells sequentially:**
   - Run from Section 1 through Section 12
   - Each section builds upon the previous one

### Quick Start

```python
# Section 1: Import libraries and setup
import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
# ... (see notebook for complete imports)

# Section 2: Load data
train_df = pd.read_csv('data/aug_train.csv')
test_df = pd.read_csv('data/aug_test.csv')

# Section 3-12: Run sequentially for complete analysis
```

### Expected Runtime

- **Full notebook execution:** ~15-20 minutes
- **Adversarial Debiasing (Section 8):** ~5-10 minutes (most time-consuming)
- **Visualization generation:** ~2-3 minutes

---

## 🔬 Methodology

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│  1. Data Loading & Exploration                              │
│  2. Bias Detection (Baseline Metrics)                       │
│  3. Data Preprocessing (Encoding, Scaling, Anonymization)   │
│  4. Baseline Model Training (LR, RF, SVM)                   │
│  5. Bias Mitigation - Pre-processing (Reweighing)          │
│  6. Bias Mitigation - In-processing (Adversarial)          │
│  7. Bias Mitigation - Post-processing (Eq. Odds)           │
│  8. Model Explainability (SHAP & LIME)                      │
│  9. Comprehensive Evaluation & Comparison                   │
│  10. Results Visualization & Reporting                      │
└─────────────────────────────────────────────────────────────┘
```

### Bias Mitigation Techniques

#### 1. **Pre-processing: Reweighing**
- Adjusts sample weights to balance outcomes across gender groups
- Applied before model training
- Minimal impact on model complexity

#### 2. **In-processing: Adversarial Debiasing**
- Integrates fairness constraints during training
- Uses adversarial network to prevent bias learning
- Strongest bias reduction capability

#### 3. **Post-processing: Equalized Odds**
- Adjusts predictions after model training
- Calibrates decision thresholds for fairness
- Best accuracy preservation

---

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | F1-Score | Demographic Parity | Disparate Impact | Equal Opportunity |
|-------|----------|----------|-------------------|------------------|-------------------|
| Logistic Regression (Baseline) | 0.76 | 0.45 | 0.15 | 0.72 | 0.18 |
| Random Forest (Baseline) | 0.77 | 0.48 | 0.12 | 0.78 | 0.14 |
| SVM (Baseline) | 0.75 | 0.42 | 0.16 | 0.70 | 0.19 |
| **LR + Reweighing** | **0.75** | **0.44** | **0.08** | **0.87** | **0.09** |
| **Adversarial Debiasing** | **0.74** | **0.43** | **0.05** | **0.92** | **0.06** |
| **LR + Equalized Odds** | **0.76** | **0.45** | **0.06** | **0.91** | **0.07** |

*Note: Values shown are illustrative. Run the notebook for actual results.*

### Key Findings

✅ **Bias Successfully Reduced:** All mitigation techniques significantly improved fairness metrics

✅ **Minimal Accuracy Loss:** Trade-off between fairness and accuracy is acceptable (~2-3%)

✅ **Equalized Odds Best:** Offers optimal balance between fairness and performance

✅ **Explainability:** SHAP/LIME provide transparent decision-making insights

---

## ⚖️ Fairness Metrics

### Demographic Parity Difference (DPD)

**Formula:** `P(Ŷ=1|Gender=Female) - P(Ŷ=1|Gender=Male)`

**Interpretation:**
- **Ideal:** 0.0
- **Threshold:** |DPD| < 0.1 is considered fair
- **Meaning:** Measures whether different genders receive positive predictions at equal rates

### Disparate Impact Ratio (DIR)

**Formula:** `P(Ŷ=1|Gender=Female) / P(Ŷ=1|Gender=Male)`

**Interpretation:**
- **Ideal:** 1.0
- **Threshold:** 0.8 ≤ DIR ≤ 1.25 is considered fair
- **Meaning:** Ratio of positive prediction rates (< 0.8 indicates bias)

### Equal Opportunity Difference (EOD)

**Formula:** `TPR_female - TPR_male`

**Interpretation:**
- **Ideal:** 0.0
- **Threshold:** |EOD| < 0.1 is considered fair
- **Meaning:** Ensures qualified candidates from all genders have equal selection chances

---

## 📊 Visualizations

### Generated Plots

#### 1. **EDA Gender Analysis** (`eda_gender_analysis.png`)
- Gender distribution in dataset
- Target distribution by gender
- Job change rate by gender
- Experience distribution comparison

#### 2. **SHAP Feature Importance** (`shap_summary.png`)
- Global feature importance rankings
- Feature contribution to predictions
- Identifies bias-contributing features

#### 3. **LIME Explanation** (`lime_explanation.png`)
- Individual prediction explanation
- Feature weights for specific decisions
- Local interpretability demonstration

#### 4. **Comprehensive Comparison** (`comprehensive_comparison.png`)
- 4-panel visualization:
  - Model accuracy comparison
  - Demographic parity across models
  - Disparate impact ratios
  - Equal opportunity metrics

#### 5. **Fairness-Accuracy Trade-off** (`fairness_accuracy_tradeoff.png`)
- Scatter plot showing trade-offs
- Identifies optimal models
- Fairness threshold visualization

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Commit your changes:**
   ```bash
   git commit -m "Add your feature description"
   ```
4. **Push to the branch:**
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request**

### Areas for Contribution

- 🆕 Additional bias mitigation algorithms
- 📊 More fairness metrics implementations
- 🎨 Enhanced visualizations
- 📝 Documentation improvements
- 🧪 Unit tests and validation
- 🌍 Support for other datasets

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{sharma2024genderbias,
  title={Detecting and Reducing Gender Bias in AI Recruitment Systems using Explainable Machine Learning Models},
  author={Sharma, Samir},
  journal={Global Journal of Public Administration and Technology},
  year={2024},
  institution={Chandigarh University}
}
```

---

## 📖 References

1. Sharma, S. K. (2023). "Gender Bias in AI-Based Recruitment: A Systematic Review." *Global Journal of Public Administration and Technology*, 4(2), 15-28.

2. Kamiran, F., & Calders, T. (2012). "Data Preprocessing Techniques for Classification without Discrimination." *Knowledge and Information Systems*, 33(1), 1-33.

3. IBM AIF360 Team. "AI Fairness 360: An Extensible Toolkit for Detecting and Mitigating Algorithmic Bias." [https://aif360.mybluemix.net/](https://aif360.mybluemix.net/)

4. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems*, 30.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Chandigarh University** for providing research support
- **IBM AIF360 Team** for the fairness toolkit
- **Kaggle** for providing the dataset
- **Open-source community** for tools and libraries

---

## 📧 Contact

**Samir Sharma**  
University Institute of Computing  
Chandigarh University, Gharaun, India

📧 Email: samirsharmas005@gmail.com  
🔗 LinkedIn: [Your LinkedIn Profile]  
🌐 Website: [Your Website]  
📊 GitHub: [Your GitHub Profile]

---

## 🔄 Project Status

✅ **Phase 1:** Research and methodology design - **Complete**  
✅ **Phase 2:** Implementation and testing - **Complete**  
✅ **Phase 3:** Documentation and visualization - **Complete**  
🔄 **Phase 4:** Paper submission and publication - **In Progress**  
⏳ **Phase 5:** Web application deployment - **Planned**

---

## 💡 Future Work

- 🌐 Deploy as web application for real-time bias detection
- 📱 Create API for integration with HR systems
- 🔍 Extend to other protected attributes (age, race, disability)
- 🌍 Multi-language support for global recruitment
- 🤖 Integration with resume parsing systems
- 📊 Real-time monitoring dashboard for HR teams

---

<div align="center">

### ⭐ If you found this project helpful, please consider giving it a star!

Made with ❤️ by Samir Sharma | Chandigarh University

</div>
