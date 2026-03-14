"""
========================================================
Retention Engine
========================================================

This module:
1. Extracts churn-driving features using SHAP values
2. Maps those features to dynamic retention strategies

Logic:
- Only positive SHAP values are considered churn drivers
- Negative SHAP values reduce churn probability and are ignored
"""

import pandas as pd


# ======================================================
# 1️⃣ Extract Top Positive SHAP Drivers
# ======================================================

def get_top_shap_reasons(shap_values, feature_names, top_n=3):
    """
    Returns top N features that positively contribute to churn.

    Parameters:
        shap_values (array): SHAP values for one customer
        feature_names (list): Feature column names
        top_n (int): Number of top features to return

    Returns:
        DataFrame: Top churn-driving features
    """

    # Create DataFrame of features and SHAP values
    df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": shap_values
    })

    # Keep only features that INCREASE churn probability
    df = df[df["SHAP_Value"] > 0]

    # Sort by highest impact first
    df = df.sort_values(by="SHAP_Value", ascending=False)

    # Return top N
    return df.head(top_n)


# ======================================================
# 2️⃣ Map Features to Retention Actions
# ======================================================

def map_rules_to_actions(top_features_df):
    """
    Maps churn-driving features to personalized retention strategies.

    Parameters:
        top_features_df (DataFrame): Output from get_top_shap_reasons()

    Returns:
        List[str]: Recommended retention actions
    """

    actions = []

    for _, row in top_features_df.iterrows():
        feature = row["Feature"]

        # Rule-based mapping
        if "age" in feature:
            actions.append("Offer age-specific benefits")

        elif "products_number" in feature or "single_product" in feature:
            actions.append("Offer bundle discount or cross-sell products")

        elif "balance" in feature:
            actions.append("Provide premium financial benefits")

        elif "active_member" in feature:
            actions.append("Increase personalized engagement outreach")

        elif "credit_score" in feature:
            actions.append("Provide financial advisory support")

        elif "country" in feature:
            actions.append("Investigate region-specific churn behavior")

        else:
            actions.append("Provide personalized retention offer")

    # Remove duplicates
    return list(set(actions))
