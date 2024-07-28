import pandas as pd
import numpy as np

class SimilarPlayerDimensions:
    """
    Class used to predict the most similar player given physical dimensions
    """
    def __init__(self, active_players: pd.DataFrame):
        self.active_players = active_players
    
    def predict_similar_player(self, user_height: int, user_weight: int, user_BMI: int, user_pos_prediction: str) -> pd.Series:
        """
        This function determines the most similar NBA player in 2023 to the physical dimensions provided
        by the user as input. It goes about this by firstly, filtering for active NBA players with the predicted
        position (G, F, C). Then, compute the cosine similiarty between those filtered active NBA players and the user
        input attributes. Finally, keep only the active NBA player with the greatest cosine similiarity score to have
        their name and image outputted in the application.

        Args:
            user_height (int): User height input
            user_weight (int): User weight input
            user_BMI (int): User bmi winput
            user_pos_prediction (str): Predicted position of user

        Returns:
            pd.Series: A pandas series containing relevant information about the most similar active NBA player.
        """
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
        
        # Find most similar player, with iloc handling multiple 'most similar' players
        most_similar_player = active_pos_players[active_pos_players['cosine_similarity'] == np.max(np.squeeze(cosine_res))].iloc[0, :]
        return most_similar_player