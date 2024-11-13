
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# Authenticate with Spotify API
def authenticate_spotify():
    client_id = "YOUR_CLIENT_ID"
    client_secret = "YOUR_CLIENT_SECRET"
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))
    return sp

# Fetch audio features for a list of track IDs
def get_audio_features(sp, track_ids):
    features = sp.audio_features(track_ids)
    features_df = pd.DataFrame([{
        'id': f['id'],
        'danceability': f['danceability'],
        'energy': f['energy'],
        'tempo': f['tempo']
    } for f in features if f is not None])
    return features_df

# K-means clustering on audio features
def cluster_tracks(features_df, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters)
    features = features_df[['danceability', 'energy', 'tempo']]
    clusters = kmeans.fit_predict(features)
    features_df['cluster'] = clusters
    return features_df, kmeans

# Recommend tracks from the same cluster
def recommend_from_cluster(features_df, kmeans, track_id):
    track = features_df[features_df['id'] == track_id]
    if track.empty:
        return "Track not found in cluster data."
    
    cluster_label = track['cluster'].values[0]
    recommendations = features_df[features_df['cluster'] == cluster_label]
    recommended_tracks = recommendations.sample(min(5, len(recommendations)))
    return recommended_tracks['id'].tolist()

# Run the recommendation system
def recommend_music():
    sp = authenticate_spotify()
    track_id = input("Enter a track ID for a recommendation: ")
    # Sample track IDs for demonstration purposes
    sample_track_ids = ["7ouMYWpwJ422jRcDASZB7P", "1v2xyoy8pJ8VXXvawNSpqH", "6F5c58TMEs1byxUstkzVeM"]
    features_df = get_audio_features(sp, sample_track_ids)
    features_df, kmeans = cluster_tracks(features_df)
    recommendations = recommend_from_cluster(features_df, kmeans, track_id)
    print("Recommended tracks based on cluster:")
    for track in recommendations:
        print(track)

if __name__ == "__main__":
    recommend_music()
