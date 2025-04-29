import numpy as np
import pandas as pd
## Import necessary modules for collaborative filtering
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
from collections import defaultdict
from collections import Counter

# Create new user-item interaction arrays
user_ids = np.array([9, 9, 9])  # Assuming the user is already encoded as 9
anime_ids = np.array([21, 136, 9253])  # These should be encoded IDs used in the original model
ratings = np.array([1.0, 1.0, 1.0])  # Normalize rating if needed (for sigmoid output)

# Reshape input arrays to match input expectations
X_finetune = [user_ids, anime_ids]
y_finetune = ratings

checkpoint_filepath = './weights/model.weights.h5'

def RecommenderNet(num_users, num_animes, embedding_size=128):
    # User input layer and embedding layer
    user = Input(name='user_encoded', shape=[1])
    user_embedding = Embedding(name='user_embedding', input_dim=num_users, output_dim=embedding_size)(user)
    
    # Anime input layer and embedding layer
    anime = Input(name='anime_encoded', shape=[1])
    anime_embedding = Embedding(name='anime_embedding', input_dim=num_animes, output_dim=embedding_size)(anime)
    
    # Dot product of user and anime embeddings
    dot_product = Dot(name='dot_product', normalize=True, axes=2)([user_embedding, anime_embedding])
    flattened = Flatten()(dot_product)
    
    # Dense layers for prediction
    dense = Dense(64, activation='relu')(flattened)
    output = Dense(1, activation='sigmoid')(dense)
    
    # Create and compile the model
    model = Model(inputs=[user, anime], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=["mae", "mse"])
    
    return model

model = RecommenderNet(num_users, num_animes)

# Load the saved weights (already done in your script)
model.load_weights(checkpoint_filepath)

# # Fine-tune on the new small batch
# model.fit(
#     x=X_finetune,
#     y=y_finetune,
#     batch_size=1,
#     epochs=5,  # You can play around here—start small
#     verbose=1
# )

user_encoder = LabelEncoder()
users_encoded = user_encoder.fit_transform(df["user_id"])
num_users = len(user_encoder.classes_)

## Encoding anime IDs
anime_encoder = LabelEncoder()
df["anime_encoded"] = anime_encoder.fit_transform(df["anime_id"])
num_animes = len(anime_encoder.classes_)



def testModel(whichones):
    user_id = 9
    user_input = np.array([])

    user2user_encoded = dict(zip(df['user_id'], df['user'].cat.codes))
    anime2anime_encoded = dict(zip(df['anime_id'], df['anime'].cat.codes))

    user_encoded2user = dict(zip(df['user'].cat.codes, df['user_id']))
    anime_encoded2anime = dict(zip(df['anime'].cat.codes, df['anime_id']))


    encoded_user = user2user_encoded[4]     # for user_id = 1
    encoded_anime = anime2anime_encoded[390]

    user_input = np.array([[encoded_user]])
    anime_input = np.array([[encoded_anime]])

    predicted_score = model.predict([user_input, anime_input])
    print("Predicted score:", predicted_score[0][0])

df_anime=pd.read_csv('/Users/joeljvarghese/Documents/Workspace/Recommendations/anime-recommendation-system/data/anime-dataset-2023.csv')

popularity_threshold = 50
df_anime= df_anime.query('Members >= @popularity_threshold')

def find_similar_animes(name, n=10, return_dist=False, neg=False):
    try:
        anime_row = df_anime[df_anime['Name'] == name].iloc[0]
        index = anime_row['anime_id']
        encoded_index = anime_encoder.transform([index])[0]
        weights = anime_weights
        dists = np.dot(weights, weights[encoded_index])
        sorted_dists = np.argsort(dists)
        n = n + 1            
        if neg:
            closest = sorted_dists[:n]
        else:
            closest = sorted_dists[-n:]
        print('Animes closest to {}'.format(name))
        if return_dist:
            return dists, closest
        
        SimilarityArr = []
        
        for close in closest:
            decoded_id = anime_encoder.inverse_transform([close])[0]
            anime_frame = df_anime[df_anime['anime_id'] == decoded_id]
            
            anime_name = anime_frame['Name'].values[0]
            english_name = anime_frame['English name'].values[0]
            name = english_name if english_name != "UNKNOWN" else anime_name
            genre = anime_frame['Genres'].values[0]
            Synopsis = anime_frame['Synopsis'].values[0]
            similarity = dists[close]
            similarity = "{:.2f}%".format(similarity * 100)
            SimilarityArr.append({"Name": name, "Similarity": similarity, "Genres": genre, "Synopsis":Synopsis})
        Frame = pd.DataFrame(SimilarityArr).sort_values(by="Similarity", ascending=False)
        return Frame[Frame.Name != name]
    except:
        print('{} not found in Anime list'.format(name))

pd.set_option('display.max_colwidth', None)

Feed = find_similar_animes('Shigatsu wa Kimi no Uso', n=5, neg=False)

for item in Feed:
