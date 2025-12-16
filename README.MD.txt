# Medical Appointment No-Show Prediction

Hello and Welcome!
I'm Eyal. This is an advanced Data Science project focused on predicting whether patients will show up for their medical appointments.

## The Business Problem
In healthcare, "No-shows" (patients who miss appointments without cancelling) are a major issue. They cause financial losses and waste doctors' valuable time.
The Challenge: The dataset is highly imbalanced - about 80% of patients show up, making it hard for standard models to detect the 20% who don't.
The Goal: Build a model that prioritizes **Recall** identifying as many "No-shows" as possible to allow proactive intervention (e.g., reminder calls).

##Methodology & Technical Approach
I built a robust pipeline to handle the class imbalance and large dataset (~110k rows):
1.  **Feature Engineering:**
    * Created `WaitingDays`: The gap between scheduling and the appointment (Crucial predictor).
    * Extracted `DayOfWeek`: To analyze if specific days have higher dropout rates.
    * **Data Cleaning:** Fixed logical errors in the dataset (e.g., negative ages, negative waiting times).
2.  **Handling Imbalance:** Used **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic examples of "No-shows", preventing the model from being biased toward the majority class.
3.  **Model Selection:** Switched from a single Decision Tree to a **Random Forest Classifier** (an ensemble of 100 trees) for better stability and generalization.

## Key Findings
Time Matters: The longer the wait between scheduling and the appointment, the higher the chance of a no-show.
The "Imbalance Trap": A naive model achieves ~80% accuracy by simply guessing "Show" for everyone, but it fails to catch a single no-show. My approach sacrificed some accuracy to gain actual predictive power.

## Results
Since the business goal is to catch missing patients, I optimized for **Recall** rather than just Accuracy.

| Metric | Score | Business Interpretation |
|--------|-------|-------------------------|
| Recall (Sensitivity) | ~76.1% | The model successfully identifies the majority of patients who would have missed their appointment. |
| Accuracy | ~62% | Lower than baseline, but reflects a model that actually takes risks to find the minority class. |
| F1 Score | ~0.44 | Represents the trade-off between precision and recall. |

## 🛠️ Tech Stack
Python: Core language.
Scikit-Learn: Random Forest, Metrics.
Imbalanced-Learn: SMOTE pipeline.
Pandas & NumPy: Advanced data manipulation.
Matplotlib & Seaborn: Data Visualization.

1. Clone this repository:
   ```bash
   git clone (https://github.com/Eyalyadai/Medical_Appointment_ML.git)

2.  Install requirements:
   ```bash
   pip install -r requirements.txt

3. Run the Notebook:
  Open notebooks/MedicalAppointment.ipynb in Jupyter Notebook.

**Created by Eyal Yadai. If you have any questions or feedback, feel free to reach out!**