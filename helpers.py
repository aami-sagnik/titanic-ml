import pandas as pd
from sklearn.feature_selection import mutual_info_classif

def find_mutual_information(df, features, target, discrete_features):
    X = df[features]
    y = df[target]
    discrete_features_mask = [ f in discrete_features for f in X.columns ]
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features_mask, random_state=42)
    return pd.Series(data=mi_scores, index=X.columns, name="MI Scores")