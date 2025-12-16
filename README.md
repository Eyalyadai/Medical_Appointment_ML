# Medical Appointment No-Show Prediction 🏥

## 📌 Project Overview
End-to-end data science project focused on predicting whether patients will miss their scheduled medical appointments ("No-shows").

Missed appointments create operational inefficiencies and financial losses for healthcare providers. This project aims to identify high-risk no-show patients in advance, enabling proactive interventions such as reminder calls or schedule optimization.

---

## 🎯 Business Problem
In the healthcare domain, appointment no-shows are a persistent challenge.
The dataset is **highly imbalanced**, with approximately **80% show-ups** and **20% no-shows**.

A naive model that predicts "Show" for every patient achieves high accuracy but provides **no real business value**.
Therefore, the primary goal of this project is to **maximize Recall**, ensuring that as many no-show cases as possible are correctly identified.

---

## 🧪 Dataset
- Real-world medical appointment dataset
- ~110,000 appointment records
- Target variable: Appointment attendance (Show / No-show)
- Includes demographic, scheduling, and historical features

---

## 🛠️ Methodology & Technical Approach

### Feature Engineering
- **WaitingDays**: Time gap between scheduling and appointment date (key predictive feature)
- **DayOfWeek**: Extracted to analyze behavioral patterns by weekday
- **Data Cleaning**:
  - Removed logically invalid values (e.g., negative ages, negative waiting times)

### Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to address severe class imbalance
- All preprocessing steps, including SMOTE, were applied **only on the training set** using a pipeline to prevent data leakage

### Modeling
- Transitioned from a single Decision Tree to a **Random Forest Classifier**
- Ensemble approach (100 trees) provided improved stability and generalization
- Model selection driven by business-oriented evaluation metrics

---

## 📊 Evaluation & Results
Given the business objective, the model was optimized for **Recall** rather than raw accuracy.

| Metric | Score | Business Interpretation |
|--------|-------|-------------------------|
| Recall (Sensitivity) | ~76% | Successfully identifies the majority of patients likely to miss their appointment |
| Accuracy | ~62% | Lower than baseline but reflects meaningful risk detection |
| F1 Score | ~0.44 | Balances recall-focused detection with prediction reliability |

A confusion matrix was used to analyze trade-offs between false positives and false negatives, prioritizing recall due to the high cost of missed appointments.

---

## 🔍 Key Insights
- **Waiting time is the strongest predictor**: longer delays significantly increase no-show probability
- **Accuracy alone is misleading** in imbalanced healthcare datasets
- Business-aligned metrics are essential for actionable machine learning solutions

---

## 🧰 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- Jupyter Notebook

---

## Let's Connect
**Built by Eyal Yadai**
(www.linkedin.com/in/eyal-yadai-13757832b)

---

## 🚀 How to Run
```bash
# 1. Clone the repository
git clone [https://github.com/Eyalyadai/Medical_Appointment_ML.git](https://github.com/Eyalyadai/Medical_Appointment_ML.git)

# 2. Install requirements
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook notebooks/MedicalAppointment.ipynb
