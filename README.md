# XGBoost
 Predicting Insurance Claim Amounts using XGBoost


 This project builds a machine learning pipeline to predict insurance claim amounts based on member demographics, policy details, and historical claims data.
It uses a synthetic dataset generated with Faker and trains an XGBoost regression model to estimate expected claims.

 Features
✅ Generates synthetic insurance claim data with realistic features (names, ages, cover amounts, dates, occupations, payment status).
✅ Encodes categorical variables and normalizes features.
✅ Builds and evaluates an XGBoost regression model to predict claim amounts.
✅ Calculates key performance metrics:

Mean Squared Error (MSE)

R² Score

 Supports predicting claim amounts for new clients based on their details.

 How it works
 Data generation
Uses Faker to generate a dataset of n=1000 insurance claims.

Each record includes:

Member name

Gender, age

Cover amount

Relationship to main member (spouse, child, etc.)

Claim date

Claim amount (based on a normal distribution adjusted by cover)

Occupation

Payment status

 Data preprocessing
Label encodes categorical variables: gender, relationship, occupation, payment_status.

Converts claim_date into days since epoch (numeric).

Scales all features with StandardScaler for robust gradient boosting.

⚙ Model training
Trains an XGBoost Regressor (XGBRegressor) with:

n_estimators=100

learning_rate=0.1

max_depth=6

Objective is squared error regression.

 Model evaluation
Calculates MSE and R² on the test data.

Predicts claim amount for a new synthetic client.


XGBoost for regression

scikit-learn: preprocessing, train-test split, metrics

Faker: generating synthetic data

Pandas / NumPy: data manipulation

 How to run
 Requirements
Install dependencies:

bash
Copy
Edit
pip install numpy pandas faker matplotlib seaborn scikit-learn xgboost
Running the model
Run the notebook or script:

bash
Copy
Edit
jupyter notebook claim_prediction_xgboost.ipynb

python claim_prediction_xgboost.py
 Example output
yaml
Copy
Edit
Mean Squared Error: 500,000.25
R^2 Score: 0.78
Predicted Claim Amount: P7,250.00
 Use case scenarios
✅ Actuaries predicting expected payouts.
✅ Insurance companies simulating future cash flows.
✅ Risk teams benchmarking reserve requirements.

 License
This project is open-source under the MIT License.


