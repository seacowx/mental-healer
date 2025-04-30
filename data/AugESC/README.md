# Guidelien for filtering the AugESC dataset

1. Run `sentiment_filter.py` to remove datum with positive sentiment.
2. Run `narrator_filter.py` to remove datum that describes events regarding others
    WANT: I experienced something that had a negative impact on myself.
    REMOVE: My friend experienced something that had a negative impact on them (but has no impact on me).
3. Run `match_persona.py` to get the top-10 persona matched profiles for each datum.


# Artifacts
- `augsec.json`: the original AugESC dataset which contains multi-turn dialogues.
- `augsec_filtered.json`: the AugESC dataset with positive dialogues removed.
- `augsec_content_filtered.json`: filtered based on `augsec_filtered.json`. Entries where the event is not about the narrator themselves or where the event does not have a profound impact at present time are removed.

`augsec_content_filtered.json` is the final dataset used for initializing GRPO training.

## Data Statistics
- `augsec.json`: 65,077 dialogues
- `augsec_filtered.json`: 52,734 dialogues
- `augsec_content_filtered.json`: 47,260 dialogues

A total of 17,817 (27.378%) dialogues are removed in the filtering process.