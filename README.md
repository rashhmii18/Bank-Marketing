# Bank Marketing Classification

## Project Overview

This project focuses on predicting whether a bank client will subscribe to a term deposit based on their personal information, banking details, and previous marketing campaign information.

The project follows a complete machine learning workflow, including data exploration, data cleaning, exploratory data analysis, preprocessing, model training, evaluation, model comparison, and deployment through a Streamlit GUI prototype.

## Dataset

The dataset used in this project is the Bank Marketing dataset from the UCI Machine Learning Repository.

Dataset Source: UCI Machine Learning Repository

Dataset Link: https://archive.ics.uci.edu/dataset/222/bank+marketing

The selected bank-full.csv dataset contains 45,211 records and 17 columns.

## Objective

The main objective of this project is to build a classification model that predicts whether a client will subscribe to a term deposit.

The target variable contains two classes:

* no - Client did not subscribe
* yes - Client subscribed

## Features

The dataset contains the following features:

* age - Age of the client
* job - Type of job
* marital - Marital status
* education - Education level
* default - Whether the client has credit in default
* balance - Average yearly balance
* housing - Whether the client has a housing loan
* loan - Whether the client has a personal loan
* contact - Type of communication contact
* day - Last contact day of the month
* month - Last contact month
* duration - Duration of the current call
* campaign - Number of contacts during the current campaign
* pdays - Number of days since the client was last contacted in a previous campaign
* previous - Number of contacts before the current campaign
* poutcome - Outcome of the previous marketing campaign
* y - Target variable

## Data Cleaning

The dataset was checked for missing values and duplicate records.

* No actual missing values were found.
* No duplicate records were found.
* The unknown values in categorical features were retained because they can represent meaningful information rather than ordinary missing values.
* The pdays value of -1 was retained because it represents clients who were not previously contacted.
* Extreme values in features such as balance, campaign, and previous were not removed without evidence that they were data errors.

The target variable was converted into binary form:

* no → 0
* yes → 1

## Exploratory Data Analysis

The target variable is highly imbalanced. Around 88.30% of clients did not subscribe, while approximately 11.70% subscribed.

Because of this imbalance, accuracy alone was not considered sufficient for evaluating the classification models.

### Categorical Feature Findings

Job distribution showed that blue-collar, management, and technician were among the largest groups in the dataset.

Subscription rates varied across job categories. Students and retired clients had relatively higher subscription rates, while blue-collar clients had a lower subscription rate.

For education, tertiary-educated clients had a higher subscription rate than primary- and secondary-educated clients.

For marital status, single clients showed a higher subscription rate than married clients in this dataset.

Clients without a housing loan had a higher subscription rate than clients with a housing loan. Similarly, clients without a personal loan showed a higher subscription rate than clients with a personal loan.

For contact type, clients contacted through cellular or telephone methods had higher subscription rates than records with an unknown contact type.

Subscription rates varied considerably across months. Some months had high subscription rates but relatively few observations, so these results were interpreted carefully.

The poutcome feature showed a strong relationship with the target. Clients with a previous successful campaign outcome had a much higher subscription rate than clients in the other categories.

### Numerical Feature Findings

The age distribution showed that most clients were in their late twenties to early fifties. Subscription rates were higher among younger clients and clients aged 60 or above compared with the middle age groups.

The balance feature was strongly right-skewed, with most clients having relatively smaller balances and a small number of clients having very large balances. Negative balances were also present in the original data.

The duration feature was right-skewed, with most calls lasting a few minutes and a small number of very long calls.

The campaign feature was also strongly right-skewed, with most clients contacted only a few times during the campaign.

The previous feature showed a similar pattern, with most clients having only a small number of previous contacts and a small number having much higher values.

The pdays feature contained many -1 values, corresponding closely with clients having no previous campaign outcome.

The numerical features generally showed weak correlations with one another. The strongest relationship was between pdays and previous, with a correlation of approximately 0.45.

## Data Preprocessing

The data was divided into training and testing sets using an 80:20 split.

Stratified splitting was used to maintain the target class distribution in both sets.

Numerical features were standardized using StandardScaler.

Categorical features were converted into numerical form using OneHotEncoder with unknown categories ignored during transformation.

The preprocessing steps were fitted only on the training data and then applied to the test data to avoid data leakage.

## Models Used

The following models were evaluated:

1. Logistic Regression
2. Logistic Regression without duration
3. Balanced Logistic Regression
4. Random Forest
5. Random Forest without duration

Class weighting was used for the Balanced Logistic Regression model to give more importance to the minority class.

## Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Confusion Matrix

ROC-AUC was used to measure the model's ability to distinguish between the two classes across different classification thresholds.

PR-AUC was also included because the target variable is imbalanced and the positive class is relatively small.

## Model Comparison

| Model                             | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
| --------------------------------- | -------: | --------: | -----: | -------: | ------: | -----: |
| Logistic Regression               |   90.12% |    64.45% | 34.78% |   45.18% |  90.56% | 54.51% |
| Logistic Regression (No Duration) |   89.33% |    66.32% | 17.86% |   28.15% |  77.17% | 41.11% |
| Balanced Logistic Regression      |   84.57% |    41.82% | 81.47% |   55.27% |  90.79% | 53.76% |
| Random Forest                     |   90.45% |    65.06% | 39.60% |   49.24% |  92.63% | 61.04% |
| Random Forest (No Duration)       |   89.54% |    63.73% | 24.57% |   35.47% |  78.88% | 42.84% |

## Model Analysis

The Random Forest model with duration achieved 90.45% accuracy, 65.06% precision, 39.60% recall, 49.24% F1-score, 92.63% ROC-AUC, and 61.04% PR-AUC.

The Balanced Logistic Regression model produced the highest recall among the tested models at 81.47%. However, this increase in recall came with lower precision and accuracy, demonstrating the trade-off between identifying more positive cases and generating more false positives.

Removing duration substantially reduced the performance of both Logistic Regression and Random Forest. This shows that duration contains substantial information about the final subscription outcome.

However, duration is the length of the current call and is only known after the call has taken place. Therefore, it would not be available when making an initial pre-call prediction.

For this reason, the no-duration models provide a more realistic evaluation for a pre-call customer-targeting scenario.

Among the no-duration models, Random Forest achieved a ROC-AUC of 78.88% and PR-AUC of 42.84%, compared with 77.17% ROC-AUC and 41.11% PR-AUC for Logistic Regression.

## Feature Importance

Feature importance from the Random Forest model with duration showed that duration was the most influential feature, followed by balance, age, and day.

Among the individual categorical features, poutcome_success had high importance, which is consistent with the higher subscription rate observed among clients with a previous successful campaign outcome.

The importance of duration was interpreted carefully because it is not available before the current call.

## Final Model for Prototype

For the practical prototype, the Random Forest model without duration was used.

This model uses information that can be available before or at the beginning of a campaign contact and therefore provides a more realistic setup for pre-call prediction.

The saved model achieved:

* Accuracy: 89.54%
* Precision: 63.73%
* Recall: 24.57%
* F1-score: 35.47%
* ROC-AUC: 78.88%
* PR-AUC: 42.84%

The model and preprocessing object were saved using Joblib:

* bank_marketing_rf.pkl
* bank_marketing_preprocessor.pkl

## GUI Prototype

A Streamlit-based GUI prototype was created separately from the main machine learning notebook.

The application allows users to enter client information such as age, job, education, account balance, loan information, contact type, campaign details, and previous campaign outcome.

The entered information is processed using the saved preprocessing object and passed to the saved Random Forest model.

The application then displays:

* Predicted subscription result
* Estimated probability of subscription

The GUI does not use duration, making it consistent with the pre-call prediction scenario.

## Project Structure

```text
Bank-Marketing-Classification/
│
├── bank_marketing.ipynb
├── app.py
├── bank_marketing_rf.pkl
├── bank_marketing_preprocessor.pkl
├── bank-full.csv
└── README.md
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook / Google Colab

## How to Run the GUI

Install the required libraries:

```bash
pip install streamlit pandas scikit-learn joblib
```

Make sure the following files are in the same folder:

```text
app.py
bank_marketing_rf.pkl
bank_marketing_preprocessor.pkl
```

Run the application using:

```bash
streamlit run app.py
```

The application will open in a web browser.

## Conclusion

This project demonstrated a complete machine learning workflow for predicting bank term-deposit subscriptions. Exploratory analysis showed that client characteristics and campaign-related information were associated with subscription behavior, while the imbalanced target made evaluation using multiple metrics important.

The experiments also showed that duration provides substantial predictive information, but it is only available after the current call. Therefore, the no-duration Random Forest model was used for the practical pre-call prototype.

The project combines data analysis, classification, model evaluation, and a working Streamlit interface into a complete machine learning application.
