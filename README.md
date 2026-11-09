# AIMLZG565 Machine Learning Assignment 2: Estimation of Obesity Levels

## a. Problem Statement
Obesity is a global health issue linked to heart diseases, diabetes, and other health problems. Accurate classification of obesity levels based on individual eating habits, physical, and demographic enables early risk identification and personalized health interventions. The goal of this project is to train, evaluate, store, and deploy five supervised machine learning models ( Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine ) to predict obesity risk across 7 multi class categories
---

## b. Dataset Description
* **Dataset Name:** Estimation of Obesity Levels Based On Eating Habits and Physical Condition
* **Dataset Source:** [UCI Machine Learning Repository (Dataset ID: 544)](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
* **Problem Type:** Multi-Class Classification (7 Class: `Insufficient_Weight`, `Normal_Weight`, `Overweight_Level_I`, `Overweight_Level_II`, `Obesity_Type_I`, `Obesity_Type_II`, `Obesity_Type_III`)
* **Instances:** 2,111 
  * **Training (80%):** 1,688
  * **Testing (20%):** 423
  * **Input Features:** 16 
  1. `Gender` : Categorical: Male / Female
  2. `Age` : Numerical: Continuous age in years
  3. `Height` : Numerical: Height in meters
  4. `Weight` : Numerical: Weight in kilograms
  5. `family_history_with_overweight` : Categorical: yes / no
  6. `FAVC` : Frequent consumption of high caloric food: yes / no
  7. `FCVC` : Frequency of consumption of vegetables: 1, 2, 3
  8. `NCP` : Number of main meals: 1, 2, 3, 4
  9. `CAEC` : Consumption of food between meals: no, Sometimes, Frequently, Always
  10. `SMOKE` : Smoking : yes / no
  11. `CH2O` Consumption of water : 1, 2, 3
  12. `SCC` Calories consumption monitoring: yes / no
  13. `FAF` : Physical activity frequency: 0 to 3
  14. `TUE` : Time using technology devices: 0, 1, 2
  15. `CALC` : Consumption of alcohol: no, Sometimes, Frequently, Always
  16. `MTRANS` : Transportation method used: Automobile, Motorbike, Bike, Public_Transportation, Walking
  * **Output Column:** `NObeyesdad`
* **Preprocessing Pipeline:**
  * Numerical features are standardized using `StandardScaler` to ensure zero mean and unit variance.
  * Categorical features are encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
  * The preprocessor is fit solely on the training partition and serialized to `model/preprocessor.joblib` to prevent any data leakage.

---

## c. Links
* **GitHub Repository Link:** [https://github.com/2026ak05681/AIMLZG565-Assignmment-2](https://github.com/2026ak05681/AIMLZG565-Assignmment-2)
* **Live Streamlit App Link:** [https://aimlzg565-assignmment-2-2026ak05681.streamlit.app](https://aimlzg565-assignmment-2-2026ak05681.streamlit.app)


## d. Models Used & Evaluation Metrics Comparison Table

All five classification models are trained on the same training data and evaluated on the test splited data (`test_data.csv`):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.8747 | 0.9837 | 0.8729 | 0.8747 | 0.8731 | 0.8539 |
| Decision Tree | 0.9173 | 0.9515 | 0.9190 | 0.9173 | 0.9177 | 0.9035 |
| KNN | 0.8534 | 0.9708 | 0.8495 | 0.8534 | 0.8456 | 0.8304 |
| Naive Bayes | 0.5083 | 0.8382 | 0.5275 | 0.5083 | 0.4607 | 0.4471 |
| Support Vector Machine | 0.9480 | 0.9909 | 0.9487 | 0.9480 | 0.9482 | 0.9393 |


## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved good performance with **87.47% Accuracy**, **0.9837 AUC**, and **0.8539 MCC** but because obesity boundaries involve non-linear interactions between height and weight, a linear model boundary cannot easily distinguish obesity classes. |
| **Decision Tree** | Delivered very good performance (**91.73% Accuracy**, **0.9515 AUC**, **0.9035 MCC**). Tree-based model worked well on continuous variables like , Weight and Height alongside categorical eating habit flags without being distorted by scaling differences. |
| **KNN** | Reached **85.34% Accuracy**, **0.9708 AUC**, and **0.8304 MCC**. While effective after standardization and one-hot encoding, high-dimensional distance computation becomes somewhat noisy across 31 transformed one-hot dimensions, slightly impairing neighbor locality. |
| **Naive Bayes** | Showed the lowest metrics (**50.83% Accuracy**, **0.8382 AUC**, **0.4471 MCC**). Gaussian Naive Bayes asuumed conditional feature independence given the class label. But correlations exist between weight, height, age, physical activity frequency, and eating habits, severely made this perform bad |
| **Support Vector Machine** | Achieved the best overall performance across all evaluated metrics: **94.80% Accuracy**, **0.9909 AUC**, **0.9487 Precision**, **0.9480 Recall**, **0.9482 F1**, and **0.9393 MCC**. The RBF kernel with regularization works best at mapping non-linear combinations of height, weight, and other indicators into cleanly separate into 7 obesity levels. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (RBF Kernel)** is the decisive overall winner. It crosses all other models with **94.80% Accuracy**, a almsot perfect **0.9909 AUC**, and **0.9393 MCC**|