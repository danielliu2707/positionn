import pandas as pd
import numpy as np
import os
import pickle

class SimilarPlayerStats:
    """
    Class used to predict the most similar player given statistics
    """
    def __init__(self, active_player_stats: pd.DataFrame):
        self.active_player_stats = active_player_stats
    
    def predict_similar_player(self, user_pts: int, user_ast: int, user_trb: int, user_stl: int, user_blk: int, user_tov: int,
                               user_pos_prediction: str) -> pd.Series: 
        """
        This function determines the most similar active NBA player to the statistics provided
        by the user as input. It goes about this by firstly, filtering for active NBA players with the predicted
        position (G, F, C). Then, compute the cosine similiarty between those filtered active NBA players and the user
        input attributes. Finally, keep only the active NBA player with the greatest cosine similiarity score to have
        their name and image outputted in the application.

        Returns:
            pd.Series: A pandas series containing relevant information about the most similar active NBA player.
        """
        # Filter for predicted position:
        active_pos_players = self.active_player_stats[self.active_player_stats['pos'] == user_pos_prediction]
        
        # Obtain only relevant attributes for comparison
        active_pos_players_num = active_pos_players[['PTS','AST','REB','STL','BLK','TOV']]
        user_features = np.array([user_pts, user_ast, user_trb, user_stl, user_blk, user_tov]).reshape(1, 6)
        
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

# Example usage
active_player_stats = pd.read_csv(os.path.join("data", "api_player_stats.csv"))
similar_player = SimilarPlayerStats(active_player_stats)
# Example inputs - excluding advanced stats
pts, ast, reb, stl, blk, tov = 30, 10, 5, 1, 1, 5
print(similar_player.predict_similar_player(pts, ast, reb, stl, blk, tov, 'G'))

# Export model
pickle.dump(similar_player, open(os.path.join("models", "similar_player_stats.pkl"), "wb"))
