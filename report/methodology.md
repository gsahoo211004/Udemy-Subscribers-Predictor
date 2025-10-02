# Methodology

## 1. Data Collection
- Dataset: Udemy Finance & Accounting courses (13K+ courses).
- File: `udemy_output_All_Finance__Accounting_p1_p626.csv`.

## 2. Data Preprocessing
- Cleaned course titles (lowercase, removed punctuation).
- Created course categories using keyword mapping.
- Handled missing values using median imputation.
- Engineered features:
  - Title length
  - Word count
  - Discount percentage
  - Keyword-based flags

## 3. Exploratory Data Analysis (EDA)
- Distribution of subscribers, ratings, and reviews.
- Relationship between price, discount, and subscribers.
- Popular categories based on title keywords.

## 4. Feature Engineering
- Numeric features: ratings, reviews, lectures, practice tests, prices.
- Categorical feature: category (One-Hot Encoding).
- Text feature: course title (TF-IDF vectorization).

## 5. Model Building
- Model: RandomForestRegressor as baseline.
- Pipeline:
  - Preprocessing with ColumnTransformer.
  - Random Forest model for regression.
- Target transformed using log(1 + subscribers) to reduce skew.

## 6. Model Evaluation
- Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R².
- Baseline Results:
  - MAE ≈ 2074
  - RMSE ≈ 6257
  - R² ≈ 0.67

## 7. Deployment
- Flask API: Accepts JSON input, returns predicted subscribers.
- Streamlit app: User-friendly UI for manual input and predictions.
