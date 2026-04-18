import pandas as pd
import os

def load_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_dir, "co2.csv")

    df = pd.read_csv(file_path)

    # keep only needed columns
    df = df[['country', 'year', 'co2']].dropna()

    return df


def get_country_data(df, country):
    return df[df['country'] == country]
