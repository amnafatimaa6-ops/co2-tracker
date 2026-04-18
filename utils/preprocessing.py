import pandas as pd

def load_data(path="data/co2.csv"):
    df = pd.read_csv(path)

    # Clean essential columns
    df = df[['country', 'year', 'co2']].dropna()

    return df


def get_country_data(df, country):
    return df[df['country'] == country]
