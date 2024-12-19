import matplotlib.pyplot as plt
import numpy as np
import wave
from os import listdir, path
import mutagen
from mutagen.easyid3 import EasyID3
import librosa

from IPython.display import Audio
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
from sklearn.model_selection import train_test_split

DURATION = 30

song_data = []

MUSIC_FOLDER = "/Users/natalie/Documents/Project/music_files"

for file_name in listdir(MUSIC_FOLDER):
    if file_name.endswith(".mp3") or file_name.endswith(".m4a"):
        file_path = path.join(MUSIC_FOLDER, file_name)
        y, sr = librosa.load(file_path, duration=DURATION)  #Load first 30 seconds
            
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()

        song_data.append({
            "File Name": file_name,
            "Spectral Centroid": spectral_centroid,
            "Zero-Crossing Rate": zero_crossing_rate,
            "Tempo": tempo
            })

song_df = pd.DataFrame(song_data)
song_df.head()
genres = []
for file_name in song_df["File Name"]:
    audio = EasyID3(os.path.join(MUSIC_FOLDER, file_name))
    genre = audio.get("genre", ["Unknown"])[0]
    genres.append(genre)

song_df["Genre"] = genres

song_df = song_df[song_df["Genre"] != "Unknown"]
# print(len(song_df))

encoder = LabelEncoder()
song_df["Genre"] = label_encoder.fit(song_df["Genre"])

features = ["Spectral Centroid", "Zero-Crossing Rate", "Tempo"]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(song_df[features])

train, test = train_test_split(df_scaled, test_size=0.5)

song_model = RandomForestClassifier()

features = train.drop(columns="Genre")
test_features = test.drop(columns="Genre")
genres = train["Genre"]
test_genres = test["Genre"]

song_model.fit(features, genres)
test_predictions = song_model.predict(test_features)

print(accuracy_score(test_features, test_genres))
print(classification_report(test_features, test_genres))

pca = PCA(n_components=4)
pca_df = pca.fit_transform(df_scaled)

#Adjust and remodel based on the graphs I can no longer see =\