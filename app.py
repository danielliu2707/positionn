import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from datetime import datetime
import os
import sklearn
from similar_player_stats import SimilarPlayerStats

# Page config
st.set_page_config(
    page_title="Positionn",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make sidebar wider
st.markdown(
    """
    <style>
        [data-testid="stSidebar"][aria-expanded="true"]{
            min-width: 350px;
            max-width: 350px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Helper functions
def get_position(predicted_pos: str, my_dict: dict):
    """Maps the Machine Learning model output (G, F, C) to their expanded names."""
    for key, value in my_dict.items():
        if value == predicted_pos:
            return key

def load_model(model):
    """Load ML model with caching."""
    return pickle.load(open(model, 'rb'))

def show_output(similar_player_id, similar_player_name, stats_df, position, position_prob):
    """Loads playstyle text, similar player and images following the position classification."""
    
    # Title with emoji
    st.title("🏀 Your Basketball Analysis")
    
    # Position prediction
    st.markdown(f"""
        <div style='text-align: center; background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <span style='font-size: 24px;'>
                You project as a <span style='color: #1f77b4; font-weight: bold;'>{position}</span>
            </span>
            <br/>
            <span style='font-size: 16px; color: #666;'>
                ({round(position_prob * 100, 1)}% probability)
            </span>
        </div>
    """, unsafe_allow_html=True)

    # Player comparison section
    st.subheader("NBA Player Comparison")
    
    # Center the player name and image using container and custom CSS
    st.markdown("""
        <style>
        .player-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            margin: 0 auto;
            max-width: 300px;
        }
        .player-name {
            color: #1f77b4;
            font-size: 24px;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="player-container">
            <div class="player-name">{similar_player_year} {similar_player_name}</div>
        </div>
    """, unsafe_allow_html=True)

    # Center the image using columns
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        try:
            st.image(f'player_headshots/{similar_player_id}.png', use_container_width=True)
        except:
            st.warning("Player headshot not available")

    # Breakdown section header - different for dimensions vs statistics
    if 'Dimensions' in stats_df.columns:
        similar_player_fname = similar_player_name.split(' ')[0]
        st.subheader(f"{similar_player_fname}'s Physical Profile")
    else:
        similar_player_fname = similar_player_name.split(' ')[0]
        st.subheader(f"{similar_player_fname}'s Statistical Breakdown")
    
    # Create clean stats table based on whether it's dimensions or statistics
    if 'Dimensions' in stats_df.columns:
        # For dimensions case
        clean_stats = pd.DataFrame({
            'Measurement': ['Height', 'Weight'],
            'Value': [
                stats_df['Dimensions'][0],  # Height
                stats_df['Dimensions'][1]   # Weight
            ]
        })
    else:
        # For statistics case
        clean_stats = pd.DataFrame({
            'Statistic': ['Points', 'Assists', 'Rebounds', 'Steals', 'Blocks', 'Turnovers'],
            'Per Game': [
                f"{stats_df['Averages'][0].split(' ')[0]} ppg",
                f"{stats_df['Averages'][1].split(' ')[0]} apg",
                f"{stats_df['Averages'][2].split(' ')[0]} rpg",
                f"{stats_df['Averages'][3].split(' ')[0]} spg",
                f"{stats_df['Averages'][4].split(' ')[0]} bpg",
                f"{stats_df['Averages'][5].split(' ')[0]} tpg"
            ]
        })

    # Center and style the table
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.table(clean_stats.style.set_properties(**{
            'text-align': 'center',
            'font-size': '16px',
            'padding': '8px'
        }))

    # Scouting report
    st.subheader("Scouting Report")
    
    if 'Dimensions' in stats_df.columns:
        st.markdown(f"""
            <div style='text-align: center; font-style: italic; color: #666; padding: 15px;'>
                Your physical measurements align closely with {similar_player_name}'s profile. 
                Players with similar builds often excel as {position.lower()}s in the NBA, utilizing their 
                physical attributes to impact the game.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style='text-align: center; font-style: italic; color: #666; padding: 15px;'>
                Your playing style closely mirrors that of {similar_player_name}. 
                Like them, you project as a {position.lower()} in the NBA, showing similar statistical patterns
                across key performance metrics.
            </div>
        """, unsafe_allow_html=True)

# Initialize position dictionary
position_dict = {"Forward": "F", "Center": "C", "Guard": "G", "Please submit either player dimensions or statistics above": -1}

# Sidebar
with st.sidebar:
    st.image("img/positionn-logo.png", width=300)
    st.markdown("\n")
    st.markdown("""Ever thought, *'If I were just a few inches taller, I could've made the NBA?'*\n\nYou're not alone — and now, you can finally find out what position you'd play if the NBA were filled with average-height people 😭
                \n Give the following a shot...
                \n🏀 Physical Position Predictor: 
                Discover your ideal NBA position and player match by scaling your physique to NBA standards from an everyday human baseline.
                
                \n\n📊 Stats Position Predictor:
                Curious how your basketball stats compare to the pros? Find out which NBA position — and which player — your playstyle most closely matches.""")
    st.markdown("---")
    st.markdown("Made with ❤️ by Daniel Liu")
    st.markdown("[GitHub](https://github.com/danielliu2707/positionn) | [LinkedIn](https://www.linkedin.com/in/daniel-liu-80693a20b/)")

# Main content
st.title("Positionn: NBA Position Predictor")

# Create tabs
tab1, tab2 = st.tabs(["Player Statistics", "Player Dimensions"])

with tab1:
    st.markdown("### Enter Player Statistics")
    current_year = datetime.now().year
    
    # Create 3 columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pts = st.number_input("Enter points/game", min_value=0.00, max_value=80.00, value=None, help="Enter points/game between 0-80")
        ast = st.number_input("Enter assists/game", min_value=0.00, max_value=30.00, value=None, help="Enter assists/game between 0-30")
    
    with col2:
        trb = st.number_input("Enter rebounds/game", min_value=0.00, max_value=40.00, value=None, help="Enter rebounds/game between 0-40")
        stl = st.number_input("Enter steals/game", min_value=0.00, max_value=20.00, value=None, help="Enter steals/game between 0-20")
    
    with col3:
        blk = st.number_input("Enter blocks/game", min_value=0.00, max_value=20.00, value=None, help="Enter blocks/game between 0-20")
        tov = st.number_input("Enter turnovers/game", min_value=0.00, max_value=30.00, value=None, help="Enter turnovers/game between 0-30")

    if st.button("Predict Position from Stats"):
        try:
            # Load models
            stats_predictor = load_model(os.path.join("models", "stats_model.sav"))
            stats_le = load_model(os.path.join("models", "stats_encoder.sav"))

            # Compute advanced stats
            ast_to, stocks, fic = round(ast/tov, 2), round(stl+blk, 2), round(pts+trb+ast+stl+blk-tov, 2)
            
            # Construct input features df with ALL columns including derived stats
            input_features = pd.DataFrame([[pts, ast, trb, stl, blk, tov, ast_to, stocks, fic]], 
                            columns=['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'AST_TO', 'STOCKS', 'FIC'])
            
            # Predict position
            predicted_pos = stats_le.inverse_transform(stats_predictor.predict(input_features))[0][0]

            # Obtain probabilities for each position
            predicted_proba = stats_predictor.predict_proba(input_features).flatten()
            all_positions = stats_le.inverse_transform([0, 1, 2])
            position_prob_dict = {pos: prob for pos, prob in zip(all_positions, predicted_proba)}
            
            # Add scroll indicator
            st.markdown(
                """
                <div style='text-align: center; color: #31333F; animation: bounce 2s infinite;'>
                    👇 Scroll down to see your results! 👇
                </div>
                <style>
                    @keyframes bounce {
                        0%, 20%, 50%, 80%, 100% {transform: translateY(0);}
                        40% {transform: translateY(-10px);}
                        60% {transform: translateY(-5px);}
                    }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            # Get similar player prediction
            similar_player_model = load_model(os.path.join("models", "similar_player_stats.pkl"))
            similar_player = similar_player_model.predict_similar_player(pts, ast, trb, stl, blk, tov, predicted_pos)
            similar_player_name = similar_player['player']
            similar_player_id = similar_player['player_id']
            similar_player_pts = np.round(similar_player['PTS'], 1)
            similar_player_ast = np.round(similar_player['AST'], 1)
            similar_player_trb = np.round(similar_player['REB'], 1)
            similar_player_stl = np.round(similar_player['STL'], 1)
            similar_player_blk = np.round(similar_player['BLK'], 1)
            similar_player_tov = np.round(similar_player['TOV'], 1)
            similar_player_year = np.round(similar_player['year'], 1)
            
            # Get final position and show output
            final_position = get_position(predicted_pos, position_dict)
            
            if final_position == "Guard":
                show_output(
                    similar_player_id=similar_player_id,
                    similar_player_name=similar_player_name,
                    stats_df=pd.DataFrame({
                        f"{similar_player_name}'s Stats": ["Points", "Assists", "Rebounds", "Steals", "Blocks", "Turnovers"],
                        "Averages": [
                            str(similar_player_pts) + ' ppg',
                            str(similar_player_ast) + ' apg',
                            str(similar_player_trb) + ' rpg',
                            str(similar_player_stl) + ' spg',
                            str(similar_player_blk) + ' bpg',
                            str(similar_player_tov) + ' tpg'
                        ]
                    }),
                    position="Guard",
                    position_prob=position_prob_dict['G']
                )
            elif final_position == "Forward":
                show_output(
                    similar_player_id=similar_player_id,
                    similar_player_name=similar_player_name,
                    stats_df=pd.DataFrame({
                        f"{similar_player_name}'s Stats": ["Points", "Assists", "Rebounds", "Steals", "Blocks", "Turnovers"],
                        "Averages": [
                            str(similar_player_pts) + ' ppg',
                            str(similar_player_ast) + ' apg',
                            str(similar_player_trb) + ' rpg',
                            str(similar_player_stl) + ' spg',
                            str(similar_player_blk) + ' bpg',
                            str(similar_player_tov) + ' tpg'
                        ]
                    }),
                    position="Forward",
                    position_prob=position_prob_dict['F']
                )
            else:
                show_output(
                    similar_player_id=similar_player_id,
                    similar_player_name=similar_player_name,
                    stats_df=pd.DataFrame({
                        f"{similar_player_name}'s Stats": ["Points", "Assists", "Rebounds", "Steals", "Blocks", "Turnovers"],
                        "Averages": [
                            str(similar_player_pts) + ' ppg',
                            str(similar_player_ast) + ' apg',
                            str(similar_player_trb) + ' rpg',
                            str(similar_player_stl) + ' spg',
                            str(similar_player_blk) + ' bpg',
                            str(similar_player_tov) + ' tpg'
                        ]
                    }),
                    position="Center",
                    position_prob=position_prob_dict['C']
                )
        except TypeError:
            st.warning("Please enter all of the player statistics")

with tab2:
    st.markdown("### Enter Player Dimensions")
    current_year = datetime.now().year
    
    # Create 2 columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        height = st.number_input("Enter height (m)", min_value=100.00, max_value=250.00, value=None, help="Enter a height between 100-250cm")
        weight = st.number_input("Enter weight (kg)", min_value=30.00, max_value=250.00, value=None, help="Enter a weight between 30-250kg")
    
    with col2:
        year_start = st.number_input("Enter year you started competitive basketball", min_value=1900, max_value=current_year,
                                    step=1, value=None, help=f"Enter a starting year between: 1900-{current_year}")
        year_end = st.number_input("Enter year you last played competitive basketball", min_value=year_start, max_value=current_year,
                                  value=None, help=f"Enter a ending year between: {year_start}-{current_year}")
    
    if st.button("Predict Position from Dimensions"):
        try:
            # Load models
            dimensions_predictor = load_model(os.path.join("models", "dimensions_model.sav"))
            dimensions_le = load_model(os.path.join("models", "dimensions_encoder.sav"))

            # Construct input features array
            input_features = (np.array([[height, weight, year_start, year_end, (weight / (height/100)**2)]]))
            
            # Predict position
            dimensions_predictor.predict(input_features)
            predicted_pos = dimensions_le.inverse_transform(dimensions_predictor.predict(input_features))[0][0]

            # Obtain probabilities for each position
            predicted_proba = dimensions_predictor.predict_proba(input_features).flatten()
            all_positions = dimensions_le.inverse_transform([0, 1, 2])
            position_prob_dict = {pos: prob for pos, prob in zip(all_positions, predicted_proba)}

            # Add scroll indicator
            st.markdown(
                """
                <div style='text-align: center; color: #31333F; animation: bounce 2s infinite;'>
                    👇 Scroll down to see your results! 👇
                </div>
                <style>
                    @keyframes bounce {
                        0%, 20%, 50%, 80%, 100% {transform: translateY(0);}
                        40% {transform: translateY(-10px);}
                        60% {transform: translateY(-5px);}
                    }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            # Get similar player prediction
            similar_player_model = load_model(os.path.join("models", "similar_player_dim.pkl"))
            similar_player = similar_player_model.predict_similar_player(height, weight, (weight / (height/100)**2), predicted_pos)
            similar_player_fname = similar_player['fname']
            similar_player_lname = similar_player['lname']
            similar_player_id = similar_player['playerid']
            similar_player_height = np.round(similar_player['height'], 2)
            similar_player_weight = np.round(similar_player['weight'], 2)
            
            # Get final position and show output
            final_position = get_position(predicted_pos, position_dict)
            
            if final_position == "Guard":
                show_output(
                    similar_player_id=similar_player_id,
                    similar_player_fname=similar_player_fname,
                    similar_player_lname=similar_player_lname,
                    stats_df=pd.DataFrame({
                        f"{similar_player_fname} {similar_player_lname}'s Stats": ["Height", "Weight"],
                        "Dimensions": [
                            str(similar_player_height) + ' cm',
                            str(similar_player_weight) + ' kg'
                        ]
                    }),
                    position="Guard",
                    position_prob=position_prob_dict['G']
                )
            elif final_position == "Forward":
                show_output(
                    similar_player_id=similar_player_id,
                    similar_player_fname=similar_player_fname,
                    similar_player_lname=similar_player_lname,
                    stats_df=pd.DataFrame({
                        f"{similar_player_fname} {similar_player_lname}'s Stats": ["Height", "Weight"],
                        "Dimensions": [
                            str(similar_player_height) + ' cm',
                            str(similar_player_weight) + ' kg'
                        ]
                    }),
                    position="Forward",
                    position_prob=position_prob_dict['F']
                )
            else:
                show_output(
                    similar_player_id=similar_player_id,
                    similar_player_fname=similar_player_fname,
                    similar_player_lname=similar_player_lname,
                    stats_df=pd.DataFrame({
                        f"{similar_player_fname} {similar_player_lname}'s Stats": ["Height", "Weight"],
                        "Dimensions": [
                            str(similar_player_height) + ' cm',
                            str(similar_player_weight) + ' kg'
                        ]
                    }),
                    position="Center",
                    position_prob=position_prob_dict['C']
                )
        except TypeError:
            st.warning("Please enter all of the physical dimensions")