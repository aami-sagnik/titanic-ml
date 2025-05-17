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

def combine_family_members(df):
    df = df.copy()
    df["FamilyMembers"] = df["SibSp"] + df["Parch"]
    return df

def encode_ordinal(df, ordinal_encoding_map):
    df = df.copy()
    for feature, order in ordinal_encoding_map.items():
        df[feature] = df[feature].map({v: i for i, v in enumerate(order)})
    return df

def encode_nominal(df, nominal_features):
    nominal_dummies = pd.get_dummies(df[nominal_features], drop_first=True)
    nominal_dummies = nominal_dummies.astype(int)
    return pd.concat([df.drop(nominal_features, axis=1), nominal_dummies], axis=1)

def apply_all(df, ordinal_encoding_map, nominal_features):
    df = transform_cabin(df)
    df = transform_fare(df)
    df = encode_ordinal(df, ordinal_encoding_map)
    df = encode_nominal(df, nominal_features)
    df = combine_family_members(df)
    return df
