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
                               user_pos_prediction: str, feature_weights: list = None) -> pd.Series: 
        """
        This function determines the most similar active NBA player to the statistics provided
        by the user as input. It goes about this by firstly, filtering for active NBA players with the predicted
        position (G, F, C). Then, compute the cosine similiarty between those filtered active NBA players and the user
        input attributes. Finally, keep only the active NBA player with the greatest cosine similiarity score to have
        their name and image outputted in the application.

        Args:
            user_pts: Points per game
            user_ast: Assists per game
            user_trb: Total rebounds per game
            user_stl: Steals per game
            user_blk: Blocks per game
            user_tov: Turnovers per game
            user_pos_prediction: Predicted position (G, F, C)
            feature_weights: List of weights for [PTS, AST, REB, STL, BLK, TOV]. Defaults to equal weights if None.

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
        
        # Apply feature weights if provided
        if feature_weights is not None:
            print("\nApplying weights:", feature_weights)
            print("Before weights - First player features:", active_pos_players_num[0])
            print("Before weights - User features:", user_features[0])
            
            weights = np.array(feature_weights)
            active_pos_players_num = active_pos_players_num * weights
            user_features = user_features * weights
            
            print("After weights - First player features:", active_pos_players_num[0])
            print("After weights - User features:", user_features[0])
        
        # Compute Euclidean distance
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(active_pos_players_num, user_features)
        
        # Convert distance to similarity score (inverse of distance)
        similarity_scores = 1 / (1 + distances)  # Adding 1 to avoid division by zero
        active_pos_players['similarity_score'] = np.squeeze(similarity_scores)
        
        # Find most similar player
        most_similar_player = active_pos_players.loc[active_pos_players['similarity_score'].idxmax()]
        return most_similar_player

if __name__ == "__main__":
    # Example usage
    active_player_stats = pd.read_csv(os.path.join("data", "api_player_stats.csv"))
    similar_player = SimilarPlayerStats(active_player_stats)
    # Example inputs - excluding advanced stats
    pts, ast, reb, stl, blk, tov = 20, 10, 10, 2, 1, 5

    def print_player_details(player_series, message=""):
        print(f"\n{message}")
        print(f"Player: {player_series['player']}")
        print(f"Your Stats:    {pts:4.1f} PTS, {ast:4.1f} AST, {reb:4.1f} REB, {stl:4.1f} STL, {blk:4.1f} BLK, {tov:4.1f} TOV")
        print(f"Player Stats:  {player_series['PTS']:4.1f} PTS, {player_series['AST']:4.1f} AST, {player_series['REB']:4.1f} REB, {player_series['STL']:4.1f} STL, {player_series['BLK']:4.1f} BLK, {player_series['TOV']:4.1f} TOV")
        print(f"Similarity Score: {player_series['similarity_score']:.3f}")
        
        # Calculate percentage differences
        pct_diff = lambda x, y: abs(x - y) / ((x + y) / 2) * 100
        print("\nPercentage differences:")
        print(f"PTS: {pct_diff(pts, player_series['PTS']):4.1f}%")
        print(f"AST: {pct_diff(ast, player_series['AST']):4.1f}%")
        print(f"REB: {pct_diff(reb, player_series['REB']):4.1f}%")
        print(f"STL: {pct_diff(stl, player_series['STL']):4.1f}%")
        print(f"BLK: {pct_diff(blk, player_series['BLK']):4.1f}%")
        print(f"TOV: {pct_diff(tov, player_series['TOV']):4.1f}%")

    # Example 1: Equal weights (default)
    result1 = similar_player.predict_similar_player(pts, ast, reb, stl, blk, tov, 'G')

    # Example 2: Heavy weight on points, more weight on assists and rebounds
    weights = [3.0, 1.0, 1.0, 0.5, 0.5, 0.5]  # Weights for [PTS, AST, REB, STL, BLK, TOV] - emphasize points, de-emphasize others
    result2 = similar_player.predict_similar_player(pts, ast, reb, stl, blk, tov, 'G', feature_weights=weights)
    print_player_details(result1, "WITHOUT WEIGHTS:")
    print_player_details(result2, "\nWITH HEAVY POINTS WEIGHT (100):")

    # Export model
    # pickle.dump(similar_player, open(os.path.join("models", "similar_player_stat.pkl"), "wb"))

    print("Exported model to models/similar_player_stats.pkl")