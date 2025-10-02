import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import preprocess_data

DATA_PATH = "data/raw/udemy_output_All_Finance__Accounting_p1_p626.csv"
MODEL_PATH = "models/udemy_subscribers_pipeline.joblib"

def main():
    # Load data
    df = pd.read_csv(DATA_PATH)
    df = preprocess_data(df)

    target = "num_subscribers"
    num_features = ['rating','num_reviews','num_published_lectures',
                    'num_published_practice_tests','discount_price__amount',
                    'price_detail__amount','Discount_Percentage',
                    'title_len','title_word_count']
    cat_features = ['category']
    text_feature = 'title'

    X = df[num_features + cat_features + [text_feature]]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)
    tfidf = TfidfVectorizer(max_features=1000)

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features),
        ('text', tfidf, text_feature)
    ], sparse_threshold=0)

    model = RandomForestRegressor(n_estimators=200, random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("R2:", r2_score(y_test, y_pred))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
