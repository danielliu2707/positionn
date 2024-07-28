import pandas as pd
import numpy as np

class SimilarPlayerDimensions:
    def __init__(self, active_players: pd.DataFrame):
        self.active_players = active_players
    
    def predict_similar_player(self, user_height: int, user_weight: int, user_BMI: int, user_pos_prediction: str):
        # Filter for predicted position:
        active_pos_players = self.active_players[self.active_players['position'] == user_pos_prediction]
        
        # Obtain only relevant attributes for comparison
        active_pos_players_num = active_pos_players[['height', 'weight', 'BMI']]
        user_features = np.array([user_height, user_weight, user_BMI]).reshape(1, 3)
        
        # Scale data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        active_pos_players_num = scaler.fit_transform(active_pos_players_num)
        user_features = scaler.transform(user_features)
        
        # Compute cosine similiarty
        from sklearn.metrics.pairwise import cosine_similarity
        cosine_res = cosine_similarity(active_pos_players_num, user_features)
        active_pos_players['cosine_similarity'] = np.squeeze(cosine_res)
        
        # Find most similar player
        most_similar_player = active_pos_players[active_pos_players['cosine_similarity'] == np.max(np.squeeze(cosine_res))]
        return most_similar_player


        