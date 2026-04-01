import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min

st.set_page_config(page_title="Spotify Mood Clustering", layout="wide")

st.title("🎧 Spotify Mood Clustering using Self-Organizing Maps (SOM)")
st.write("Upload your Spotify feature dataset to get SOM-based cluster + mood labels.")

# -------------------------------
# 🔹 LOAD TRAINED SOM MODEL
# -------------------------------
MODEL_PATH = "som_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        som, neuron_kmeans, scaler = pickle.load(f)
    st.success("Model Loaded Successfully!")
except Exception as e:
    st.error(f"Failed to load SOM model: {e}")
    st.stop()

# -----------------------------------
# 🔹 REQUIRED FEATURE COLUMNS
# -----------------------------------
FEATURES = [
    "danceability","energy","valence","tempo","acousticness",
    "instrumentalness","speechiness","loudness","liveness"
]

st.sidebar.title("Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV with Spotify Audio Features", type=["csv"])

# -----------------------------------
# 🔹 PROCESS UPLOADED CSV
# -----------------------------------
# -------------------------------
# UPLOAD + ROBUST FEATURE ALIGNMENT (REPLACE existing 'if uploaded:' block)
# -------------------------------
if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # 1) Determine expected feature names from the scaler (if available)
    expected = None
    if hasattr(scaler, "feature_names_in_"):
        expected = list(getattr(scaler, "feature_names_in_"))
        st.info(f"Scaler expects these feature names (and order): {expected}")
    else:
        # Fallback: use your declared FEATURES list (ensure it matches scaler training)
        expected = [
            "danceability","energy","valence","tempo","acousticness",
            "instrumentalness","speechiness","loudness","liveness"
        ]
        st.warning("Scaler does not expose feature_names_in_; using fallback expected list. "
                   "If this is wrong, retrain scaler or update this list.")

    # 2) Make sure dataframe has all expected columns; if missing, fill with sensible defaults
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        st.warning(f"Uploaded CSV is missing columns required by scaler: {missing_cols}")
        # Fill missing by using per-column median if any numeric columns exist, else 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in missing_cols:
            if numeric_cols:
                fill_val = df[numeric_cols].median().median()  # a generic numeric fallback
            else:
                fill_val = 0.0
            df[c] = fill_val
        st.info("Missing columns added and filled with medians/zeros.")

    # 3) Reorder columns exactly in the order scaler expects
    try:
        X_df = df[expected].copy()
    except Exception as e:
        st.error("Failed to subset/reorder DataFrame to scaler's expected columns: " + str(e))
        st.stop()

    # 4) Final cleaning: replace infinities and fill any NaNs with column medians
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    for col in X_df.columns:
        if X_df[col].isna().any():
            X_df[col] = X_df[col].fillna(X_df[col].median() if X_df[col].dtype.kind in "biufc" else 0)

    # 5) Attempt to transform with scaler (now that order matches)
    try:
        X = scaler.transform(X_df)  # this should no longer raise the feature-names error
    except Exception as e:
        # if transform still fails, print diagnostic info and stop
        st.error("Scaler.transform failed despite reordering. Diagnostic: " + str(e))
        if hasattr(scaler, "feature_names_in_"):
            st.write("Scaler.feature_names_in_:", list(scaler.feature_names_in_))
            st.write("Provided dataframe columns:", list(X_df.columns))
        st.stop()

    # 6) Compute BMUs from SOM and predict cluster labels
    bmu = np.array([som.winner(x) for x in X])
    # try to predict using neuron_kmeans. It may expect 2D coords or flattened indices.
    try:
        cluster_labels = neuron_kmeans.predict(bmu.reshape(-1, 2))
    except Exception:
        # fallback to flattened neuron index input
        ncols = som._weights.shape[1]
        bmu_flat = bmu[:, 0] * ncols + bmu[:, 1]
        # neuron_kmeans might have been trained on flattened indexes; attempt predict
        try:
            cluster_labels = neuron_kmeans.predict(bmu_flat.reshape(-1, 1))
        except Exception:
            # last fallback: predict using neuron coordinates directly if supported
            cluster_labels = np.zeros(len(bmu), dtype=int)
            st.warning("Could not use neuron_kmeans to predict clusters with usual formats; returning zeros. "
                       "If this persists, confirm how neuron_kmeans was trained and saved.")

    # 7) Attach results to original dataframe and continue
    df["som_x"] = bmu[:, 0]
    df["som_y"] = bmu[:, 1]
    df["cluster"] = cluster_labels

    st.subheader("🎯 SOM Cluster Output")
    st.dataframe(df.head(20))

    # proceed with mood mapping, plotting, download, etc. (as in your original app)
    def assign_mood(row):
        if row["valence"] > 0.6 and row["danceability"] > 0.5:
            return "Happy"
        if row["energy"] > 0.7:
            return "Energetic"
        if row["acousticness"] > 0.6:
            return "Calm"
        if row["valence"] < 0.3:
            return "Sad"
        return "Chill"

    df["mood"] = df.apply(assign_mood, axis=1)

    st.subheader("🎵 Mood-Assigned Songs")
    st.dataframe(df[["track_name","artist_name","cluster","mood"]].head(25))

    st.download_button(
        "⬇️ Download Results",
        df.to_csv(index=False).encode("utf-8"),
        file_name="spotify_mood_results.csv",
    )

    # U-matrix and SOM Grid plotting (keep your existing code)
    umat = som.distance_map().T
    fig, ax = plt.subplots(figsize=(7,7))
    plt.title("U-Matrix")
    plt.imshow(umat, cmap="bone_r")
    plt.colorbar(label="Distance (Lower = Similar Neurons)")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(7,7))
    plt.title("SOM Grid — Song Mappings")
    plt.imshow(umat, cmap="bone_r")
    plt.scatter(bmu[:,1], bmu[:,0], s=20, c=cluster_labels, cmap="tab20", alpha=0.9)
    st.pyplot(fig2)

    # ----------------------------------------------------
    # 🔥 VISUALIZE THE U-MATRIX
    # ----------------------------------------------------
    st.subheader("📡 SOM U-Matrix (Cluster Separation Visualization)")
    umat = som.distance_map().T  # transpose for plotting

    fig, ax = plt.subplots(figsize=(7,7))
    plt.title("U-Matrix")
    plt.imshow(umat, cmap="bone_r")
    plt.colorbar(label="Distance (Lower = Similar Neurons)")
    st.pyplot(fig)

    # ----------------------------------------------------
    # 🔥 VISUALIZE SOM GRID WITH POINTS
    # ----------------------------------------------------
    st.subheader("🧠 SOM Grid with Mapped Songs")

    fig2, ax2 = plt.subplots(figsize=(7,7))
    plt.title("SOM Grid — Song Mappings")

    plt.imshow(umat, cmap="bone_r")
    plt.scatter(bmu[:,1], bmu[:,0], s=20, c=cluster_labels, cmap="tab20", alpha=0.9)

    st.pyplot(fig2)


# -------------------------------
# 🔹 FOOTER
# -------------------------------
st.markdown("---")
st.info("Built using SOM + KMeans neuron clustering. Upload new songs anytime!")
