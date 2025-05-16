import numpy as np
import pandas as pd

def transform_cabin(df):
    df = df.copy()
    df["Cabin"] = df["Cabin"].apply(lambda x: x[0] if x is not np.nan else "None")
    return df

def transform_fare(df):
    df = df.copy()
    df["Fare"] = np.log1p(df["Fare"])
    return df

ordinal_encoding = {
   "Pclass": [3, 2, 1]
}

def encode_ordinal(df):
    df = df.copy()
    for feature, order in ordinal_encoding.items():
        df[feature] = df[feature].map({v: i for i, v in enumerate(order)})
    return df

target = "Survived"
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked", "Fare", "Cabin"]
nominal_features = ['Sex', 'Embarked', 'Cabin']

def encode_nominal(df):
    nominal_dummies = pd.get_dummies(df[nominal_features], drop_first=True)
    nominal_dummies = nominal_dummies.astype(int)
    if target in df:
        return pd.concat([df[features].drop(nominal_features, axis=1), nominal_dummies, df[[target]]], axis=1) # for training dataset
    else:
        return pd.concat([df[features].drop(nominal_features, axis=1), nominal_dummies], axis=1) # for testing dataset

def apply_all(df):
    df = transform_cabin(df)
    df = transform_fare(df)
    df = encode_ordinal(df)
    df = encode_nominal(df)
    return df
