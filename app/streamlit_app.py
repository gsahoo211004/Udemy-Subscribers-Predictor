import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/udemy_subscribers_pipeline.joblib")

st.title("📊 Udemy Subscribers Predictor")

title = st.text_input("Course Title", "the complete sql bootcamp")
rating = st.slider("Rating", 0.0, 5.0, 4.5, 0.1)
num_reviews = st.number_input("Number of Reviews", 0, 200000, 100)
num_published_lectures = st.number_input("Number of Lectures", 0, 5000, 50)
num_published_practice_tests = st.number_input("Practice Tests", 0, 100, 0)
discount_price = st.number_input("Discounted Price ($)", 0.0, 2000.0, 10.0)
price_detail = st.number_input("Original Price ($)", 0.0, 2000.0, 100.0)
category = st.selectbox("Category", ["Business","Finance","Database","Data Science",
                                    "Data Visualization","Other"])

if st.button("Predict Subscribers"):
    input_df = pd.DataFrame([{
        'title': title,
        'rating': rating,
        'num_reviews': num_reviews,
        'num_published_lectures': num_published_lectures,
        'num_published_practice_tests': num_published_practice_tests,
        'discount_price__amount': discount_price,
        'price_detail__amount': price_detail,
        'Discount_Percentage': (price_detail - discount_price)/price_detail*100 if price_detail>0 else 0,
        'title_len': len(title),
        'title_word_count': len(title.split()),
        'category': category
    }])
    pred = model.predict(input_df)[0]
    st.success(f"Predicted Subscribers: {int(round(pred))}")
