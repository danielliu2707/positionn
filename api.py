from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import os
import sys
from similar_player_stats import SimilarPlayerStats
from similar_player_dimensions import SimilarPlayerDimensions

app = FastAPI(title="Positionn API", description="NBA Position Prediction API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom unpickler to handle classes that were pickled in __main__ context
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            if name == "SimilarPlayerStats":
                return SimilarPlayerStats
            elif name == "SimilarPlayerDimensions":
                return SimilarPlayerDimensions
        return super().find_class(module, name)

# Helper function to load models
def load_model(model_path):
    """Load ML model with pickle."""
    with open(model_path, 'rb') as f:
        return CustomUnpickler(f).load()

# Position mapping dictionary
position_dict = {"Forward": "F", "Center": "C", "Guard": "G"}

def get_position(predicted_pos: str, my_dict: dict):
    """Maps the Machine Learning model output (G, F, C) to their expanded names."""
    for key, value in my_dict.items():
        if value == predicted_pos:
            return key

# Pydantic models for request validation
class StatsInput(BaseModel):
    points: float
    rebounds: float
    assists: float
    steals: float
    blocks: float
    turnovers: float

class DimensionsInput(BaseModel):
    height_cm: float
    weight_kg: float
    year_start: int
    year_end: int

@app.get("/")
def read_root():
    """Root endpoint for API health check."""
    return {"message": "Positionn API is running", "version": "1.0.0"}

@app.post("/predict/stats")
def predict_from_stats(payload: StatsInput):
    """
    Predict NBA position and similar player based on statistics.
    
    Input: Points, rebounds, assists, steals, blocks, turnovers per game
    Output: Predicted position, probability, similar player information and stats
    """
    try:
        # Load models
        stats_predictor = load_model(os.path.join("models", "stats_model.sav"))
        stats_le = load_model(os.path.join("models", "stats_encoder.sav"))
        similar_player_model = load_model(os.path.join("models", "similar_player_stat.pkl"))
        
        pts = payload.points
        ast = payload.assists
        trb = payload.rebounds
        stl = payload.steals
        blk = payload.blocks
        tov = payload.turnovers
        
        # Compute advanced stats
        ast_to = round(ast/tov, 2) if tov > 0 else 0
        stocks = round(stl+blk, 2)
        fic = round(pts+trb+ast+stl+blk-tov, 2)
        
        # Construct input features df with ALL columns including derived stats
        input_features = pd.DataFrame([[pts, ast, trb, stl, blk, tov, ast_to, stocks, fic]], 
                        columns=['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'AST_TO', 'STOCKS', 'FIC'])
        
        # Predict position
        predicted_pos = stats_le.inverse_transform(stats_predictor.predict(input_features))[0][0]
        
        # Obtain probabilities for each position
        predicted_proba = stats_predictor.predict_proba(input_features).flatten()
        all_positions = stats_le.inverse_transform([0, 1, 2])
        position_prob_dict = {pos: float(prob) for pos, prob in zip(all_positions, predicted_proba)}
        
        # Get similar player prediction
        similar_player = similar_player_model.predict_similar_player(
            pts, ast, trb, stl, blk, tov, predicted_pos, 
            feature_weights=[3.0, 1.0, 1.0, 0.5, 0.5, 0.5]
        )
        
        # Get final position name
        final_position = get_position(predicted_pos, position_dict)
        
        # Format response
        return {
            "position": final_position,
            "position_code": predicted_pos,
            "probability": float(position_prob_dict[predicted_pos]),
            "probabilities": position_prob_dict,
            "twin": {
                "player_id": int(similar_player['player_id']),
                "name": str(similar_player['player']),
                "year": float(similar_player['year'])
            },
            "twin_stats": {
                "points": float(np.round(similar_player['PTS'], 1)),
                "assists": float(np.round(similar_player['AST'], 1)),
                "rebounds": float(np.round(similar_player['REB'], 1)),
                "steals": float(np.round(similar_player['STL'], 1)),
                "blocks": float(np.round(similar_player['BLK'], 1)),
                "turnovers": float(np.round(similar_player['TOV'], 1))
            }
        }
    except ZeroDivisionError:
        raise HTTPException(status_code=400, detail="Turnovers cannot be zero when calculating assist-to-turnover ratio")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing prediction: {str(e)}")

@app.post("/predict/dimensions")
def predict_from_dimensions(payload: DimensionsInput):
    """
    Predict NBA position and similar player based on physical dimensions.
    
    Input: Height (cm), weight (kg), year started playing, year last played
    Output: Predicted position, probability, similar player information and dimensions
    """
    try:
        # Load models
        dimensions_predictor = load_model(os.path.join("models", "dimensions_model.sav"))
        dimensions_le = load_model(os.path.join("models", "dimensions_encoder.sav"))
        similar_player_model = load_model(os.path.join("models", "similar_player_dim.pkl"))
        
        height = payload.height_cm
        weight = payload.weight_kg
        year_start = payload.year_start
        year_end = payload.year_end
        
        # Compute BMI
        bmi = weight / ((height/100)**2)
        
        # Construct input features array
        input_features = np.array([[height, weight, year_start, year_end, bmi]])
        
        # Predict position
        predicted_pos = dimensions_le.inverse_transform(dimensions_predictor.predict(input_features))[0][0]
        
        # Obtain probabilities for each position
        predicted_proba = dimensions_predictor.predict_proba(input_features).flatten()
        all_positions = dimensions_le.inverse_transform([0, 1, 2])
        position_prob_dict = {pos: float(prob) for pos, prob in zip(all_positions, predicted_proba)}
        
        # Get similar player prediction
        similar_player = similar_player_model.predict_similar_player(height, weight, bmi, predicted_pos)
        
        # Get final position name
        final_position = get_position(predicted_pos, position_dict)
        
        # Format response
        return {
            "position": final_position,
            "position_code": predicted_pos,
            "probability": float(position_prob_dict[predicted_pos]),
            "probabilities": position_prob_dict,
            "twin": {
                "player_id": int(similar_player['playerid']),
                "first_name": str(similar_player['fname']),
                "last_name": str(similar_player['lname']),
                "full_name": f"{similar_player['fname']} {similar_player['lname']}"
            },
            "twin_dimensions": {
                "height_cm": float(np.round(similar_player['height'], 2)),
                "weight_kg": float(np.round(similar_player['weight'], 2))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
