import pandas as pd
import numpy as np



df = pd.read_csv('dataset.csv')


threshold = len(df) * 0.5
df = df.dropna(axis=1, thresh=threshold)

for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        df[column].fillna(df[column].mean(), inplace=True)
    else:
        df[column].fillna(df[column].mode()[0], inplace=True)

df.drop_duplicates(inplace=True)

df['date_column'] = pd.to_datetime(df['date_column'], errors='coerce')

from scipy import stats
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
df[(z_scores < 3).all(axis=1)]  # Keep rows where z-score is less than 3 for all numeric columns

df.to_csv('cleaned_data.csv', index=False)

print("Data cleaning complete! Cleaned dataset saved as 'cleaned_data.csv'.")
