def compute_retention_impact(probability, actions, top_drivers):
    reduction = 0

    for driver in top_drivers:
        feature = driver["feature"]
        impact = driver["impact"]

        if "balance" in feature:
            reduction += impact * 0.4

        elif "products_number" in feature:
            reduction += impact * 0.3

        elif "active_member" in feature:
            reduction += impact * 0.5

        else:
            reduction += impact * 0.2

    reduction = min(reduction, 0.5)

    after = max(probability - reduction, 0)

    return {
        "before": round(probability, 4),
        "after": round(after, 4),
        "improvement": round(probability - after, 4)
    }