import re
import pandas as pd
import numpy as np

def clean_title(t):
    t = str(t).lower().strip()
    t = re.sub(r'[^\w\s]', '', t)
    return t

def categorize_title(title):
    tl = title.lower()
    if any(k in tl for k in ['sql','mysql','database']):
        return 'Database'
    if any(k in tl for k in ['tableau','power bi','data viz']):
        return 'Data Visualization'
    if 'excel' in tl or 'spreadsheet' in tl:
        return 'Spreadsheet'
    if any(kw in tl for kw in ['agile','scrum','pmp','project management']):
        return 'Project Management'
    if any(kw in tl for kw in ['financial','finance','accounting']):
        return 'Finance'
    if 'mba' in tl or 'business' in tl:
        return 'Business'
    if any(kw in tl for kw in ['write','writing','editorial']):
        return 'Writing'
    if 'sale' in tl or 'marketing' in tl:
        return 'Sales/Marketing'
    if any(kw in tl for kw in ['data science','analytics','machine learning','ml']):
        return 'Data Science'
    if 'management' in tl:
        return 'Management'
    if 'leadership' in tl:
        return 'Leadership'
    if 'communication' in tl:
        return 'Communication'
    return 'Other'

def preprocess_data(df):
    # Clean title
    df['title'] = df['title'].apply(clean_title)

    # Create category
    df['category'] = df['title'].apply(categorize_title)

    # Handle missing values
    df.fillna({'num_reviews':0,'rating':df['rating'].median()}, inplace=True)

    # Discount percentage
    if 'price_detail__amount' in df.columns and 'discount_price__amount' in df.columns:
        df['Discount_Percentage'] = np.where(
            df['price_detail__amount']>0,
            (df['price_detail__amount'] - df['discount_price__amount'])/df['price_detail__amount']*100,
            0
        )
    else:
        df['Discount_Percentage'] = 0.0

    # Title features
    df['title_len'] = df['title'].str.len()
    df['title_word_count'] = df['title'].str.split().str.len()

    return df
