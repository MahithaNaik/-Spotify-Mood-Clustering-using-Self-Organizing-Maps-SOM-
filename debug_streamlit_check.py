# debug_streamlit_check.py
from pathlib import Path
import json, pandas as pd
print("cwd:", Path.cwd())
d = Path("./spotify_mood_output")
print("spotify_mood_output exists?", d.exists())
if d.exists():
    for name in ["spotify_tracks_with_cluster_mood.csv","cluster_summary.csv","cluster_mood_scores.csv","cluster_mood_mapping.json"]:
        p = d / name
        print(name, "exists:", p.exists(), "size:", p.stat().st_size if p.exists() else "N/A")
    try:
        df = pd.read_csv(d/"spotify_tracks_with_cluster_mood.csv")
        print("Tracks rows:", len(df))
        print("Example:", df[['track_name','artist_name','cluster','mood']].head(3).to_dict(orient='records'))
    except Exception as e:
        print("Failed reading CSV:", e)
else:
    print("DATA FOLDER MISSING")
