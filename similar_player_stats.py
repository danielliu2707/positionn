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
                               user_pos_prediction: str, feature_weights: list = None, top_n: int = 10) -> pd.DataFrame: 
        """
        This function determines the most similar active NBA players to the statistics provided
        by the user as input. It goes about this by firstly, filtering for active NBA players with the predicted
        position (G, F, C). Then, compute the euclidean distance between those filtered active NBA players and the user
        input attributes. Finally, return the top N most similar players ranked by similarity score.

        Args:
            user_pts: Points per game
            user_ast: Assists per game
            user_trb: Total rebounds per game
            user_stl: Steals per game
            user_blk: Blocks per game
            user_tov: Turnovers per game
            user_pos_prediction: Predicted position (G, F, C)
            feature_weights: List of weights for [PTS, AST, REB, STL, BLK, TOV]. Defaults to equal weights if None.
            top_n: Number of top similar players to return (default: 10)

        Returns:
            pd.DataFrame: A pandas DataFrame containing the top N most similar players, sorted by similarity_score (descending).
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
        active_pos_players = active_pos_players.copy()  # Avoid SettingWithCopyWarning
        active_pos_players['similarity_score'] = np.squeeze(similarity_scores)
        
        # Ensure distinct player_id's by keeping only the entry with the HIGHEST similarity score for each player
        # This means if a player appears in multiple years, we keep the year where they were MOST similar
        if 'player_id' in active_pos_players.columns:
            # Sort by similarity_score DESCENDING (highest similarity first)
            # Then drop duplicates, keeping the FIRST occurrence (which is the highest similarity)
            # This ensures each player_id appears only once, with their most similar year
            active_pos_players = active_pos_players.sort_values('similarity_score', ascending=False)
            active_pos_players = active_pos_players.drop_duplicates(subset=['player_id'], keep='first')
            # Reset index after dropping duplicates
            active_pos_players = active_pos_players.reset_index(drop=True)
        
        # Sort by similarity score (descending) and return top N distinct players
        # Each player appears once, with their most similar year
        top_players = active_pos_players.nlargest(top_n, 'similarity_score')
        return top_players

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

    # Example 1: Equal weights (default) - get top 10
    result1 = similar_player.predict_similar_player(pts, ast, reb, stl, blk, tov, 'G', top_n=10)
    print("\nTOP 10 WITHOUT WEIGHTS:")
    for idx, (_, player) in enumerate(result1.iterrows(), 1):
        print(f"\n{idx}. {player['player']} - Similarity: {player['similarity_score']:.3f}")
        print(f"   Stats: {player['PTS']:.1f} PTS, {player['AST']:.1f} AST, {player['REB']:.1f} REB, {player['STL']:.1f} STL, {player['BLK']:.1f} BLK, {player['TOV']:.1f} TOV")

    # Example 2: Heavy weight on points, more weight on assists and rebounds - get top 10
    weights = [3.0, 1.0, 1.0, 0.5, 0.5, 0.5]  # Weights for [PTS, AST, REB, STL, BLK, TOV] - emphasize points, de-emphasize others
    result2 = similar_player.predict_similar_player(pts, ast, reb, stl, blk, tov, 'G', feature_weights=weights, top_n=10)
    print("\n\nTOP 10 WITH HEAVY POINTS WEIGHT:")
    for idx, (_, player) in enumerate(result2.iterrows(), 1):
        print(f"\n{idx}. {player['player']} - Similarity: {player['similarity_score']:.3f}")
        print(f"   Stats: {player['PTS']:.1f} PTS, {player['AST']:.1f} AST, {player['REB']:.1f} REB, {player['STL']:.1f} STL, {player['BLK']:.1f} BLK, {player['TOV']:.1f} TOV")
    
    # Example: Get just the top player (backward compatibility)
    top_player = result1.iloc[0]
    print_player_details(top_player, "\n\nMOST SIMILAR PLAYER (top 1):")

    # Export SimilarPlayerStats model (i.e. class object with the dataframe of all active NBA
    # players and the predict_similar_player method) - Based on the points weighted model
    ## pickle.dump(similar_player, open(os.path.join("models", "similar_player_stat.pkl"), "wb"))

    print("Exported model to models/similar_player_stats.pkl")