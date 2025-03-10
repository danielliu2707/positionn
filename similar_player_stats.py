import pandas as pd
import numpy as np
import os

class SimilarPlayerStats:
    """
    Class used to predict the most similar player given statistics
    """
    def __init__(self, active_players: pd.DataFrame):
        self.active_players = active_players
    
    def predict_similar_player(self, user_pts: int, user_ast: int, user_trb: int, user_stl: int, user_blk: int,
                               user_pos_prediction: str) -> pd.Series:
        """
        This function determines the most similar NBA player in 2023 to the statistics provided
        by the user as input. It goes about this by firstly, filtering for active NBA players with the predicted
        position (G, F, C). Then, compute the cosine similiarty between those filtered active NBA players and the user
        input attributes. Finally, keep only the active NBA player with the greatest cosine similiarity score to have
        their name and image outputted in the application.

        Args:
            user_pts (int): _description_
            user_ast (int): _description_
            user_trb (int): _description_
            user_stl (int): _description_
            user_blk (int): _description_
            user_pos_prediction (str): _description_

        Returns:
            pd.Series: A pandas series containing relevant information about the most similar active NBA player.
        """
        # Filter for predicted position:
        active_pos_players = self.active_players[self.active_players['position'] == user_pos_prediction]
        
        # Obtain only relevant attributes for comparison
        active_pos_players_num = active_pos_players[['pts', 'ast', 'trb', 'stl', 'blk']]
        user_features = np.array([user_pts, user_ast, user_trb, user_stl, user_blk]).reshape(1, 5)
        
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

active_player_statistics = pd.read_csv(os.path.join("data", "active_player_stats.csv"))
similar_player = SimilarPlayerStats(active_player_statistics)
print(similar_player.predict_similar_player(20, 10, 5, 1, 1, 'G'))
