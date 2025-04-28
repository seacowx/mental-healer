# Guidelien for filtering the AugESC dataset

1. Run `sentiment_filter.py` to remove datum with positive sentiment.
2. Run `narrator_filter.py` to remove datum that describes events regarding others
    WANT: I experienced something that had a negative impact on myself.
    REMOVE: My friend experienced something that had a negative impact on them (but has no impact on me).