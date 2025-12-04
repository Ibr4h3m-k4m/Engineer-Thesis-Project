# 🚗 VeReMi Dataset Classification - Engineer Thesis Project

Quick Note : This ReadMe file is a quick spin up using claude , it still needs to be correctly Updated to matchh exactly what i've done in the Project.

## 📊 Project Overview

This project tackles the challenge of detecting malicious vehicular network traffic using the VeReMi Extension Dataset. Through advanced machine learning techniques and metaheuristic optimization, we achieved **94% accuracy** and **90% macro F1-score** in classifying vehicle-to-vehicle communication attacks.

## 🎯 Key Achievements

- **94% Accuracy** on multi-class attack detection
- **90% Macro F1-Score** across all attack categories
- Balanced class distribution through intelligent grouping
- Optimized feature space using Battle Royale Optimization
- Fine-tuned hyperparameters for maximum performance

## 🏗️ Project Structure

```
├── battle_royal_optimizer_feature_selection.py  # BRO algorithm for feature selection
├── final-pfe-model-notebook.ipynb               # Main model training notebook
├── hyper-params-optimization/                   # Hyperparameter tuning experiments
├── data_File_Codes/                             # Data processing scripts
├── webapp/                                       # Web application interface
└── readme.md                                     # This file
```

## 🔬 Methodology

### 1. **Data Preprocessing**
- Removed non-essential columns to reduce dimensionality
- Addressed severe class imbalance in the original dataset
- Grouped 19 non-benign attack types into 5 meaningful categories
- Applied normalization and encoding techniques

### 2. **Feature Selection**
- Implemented **Battle Royale Optimization (BRO)** metaheuristic algorithm
- Reduced feature space while maintaining model performance
- Selected most discriminative features for attack detection

### 3. **Model Development**
Explored multiple approaches:
- **Machine Learning**: Traditional classifiers (RF, XGBoost, SVM, etc.)
- **Deep Learning**: Neural network architectures
- **Ensemble Methods**: Voting, stacking, and boosting
- **One-vs-Rest**: Multi-class classification strategies

### 4. **Hyperparameter Optimization**
- Systematic tuning of model parameters
- Grid search and random search techniques
- Cross-validation for robust evaluation

## 📈 Results

| Metric | Score |
|--------|-------|
| Accuracy | **94%** |
| Macro F1-Score | **90%** |
| Precision | High across all classes |
| Recall | Balanced detection rates |

## 🛠️ Technologies Used

- **Python** - Core programming language
- **Scikit-learn** - Machine learning models
- **TensorFlow/Keras** - Deep learning (if applicable)
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Visualization
- **Jupyter Notebook** - Experimentation and analysis

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Notebook
```bash
jupyter notebook final-pfe-model-notebook.ipynb
```

### Feature Selection
```bash
python battle_royal_optimizer_feature_selection.py
```

## 📊 Dataset

**VeReMi Extension Dataset** - A comprehensive dataset for vehicular network misbehavior detection containing:
- Benign traffic patterns
- 19 different attack types (grouped into 5 categories)
- Real-world V2V communication scenarios

> **Note**: Due to size constraints, the dataset is not included in this repository. Please download it separately and place it in `data_File_Codes/`.

## 🎓 Academic Context

This project was completed as part of my **Engineer Thesis** at [Your University]. It explores the intersection of cybersecurity, vehicular networks, and artificial intelligence.

## 📝 Future Work

- Real-time attack detection system
- Additional metaheuristic algorithms comparison
- Model deployment and API development
- Explainability analysis (SHAP, LIME)

## 👤 Author

**Ibrahim Kamraoui**


## 🙏 Acknowledgments

Special thanks to myself for the perseverance in a project which i had 0 knowledge at the start.

---

*This project demonstrates the power of combining metaheuristic optimization with machine learning for robust cybersecurity solutions in vehicular networks.*