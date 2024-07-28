import streamlit as st
import random
import numpy as np
import os
import pickle
from datetime import datetime
import base64
from similar_player_dimensions import SimilarPlayerDimensions

st.set_page_config(
    page_title = "positionn",
    page_icon = ":basketball:",
    initial_sidebar_state="collapsed"
)

# Cetner images when expanded
st.markdown(
    """
    <style>
        button[title^=Exit]+div [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Hide top multicolor header
hide_decoration_bar_style = '''<style>header {visibility: hidden;}</style>'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

# Define logo 
st.image(os.path.join("img", "positionn-logo.png"))
st.markdown("Determine what NBA position best suits you based on your Physical Dimensions or Statistics & scroll down to find out which NBA player is most like you.")

def get_position(predicted_pos: str, my_dict: dict):
    """
    Maps the Machine Learning model output (G, F, C) to their expanded names (Guard, Forward, Center).

    Args:
        predicted_pos (str): The machine learning output (G, F, C)
        my_dict (dict): The mapping of model output to expanded names

    Returns:
        str: The expanded position name
    """
    for key, value in my_dict.items():
        if value == predicted_pos:
            return key

position_dict = {"Forward": "F", "Center": "C", "Guard": "G", "Please submit either player dimensions or statistics above": -1}

# Loading in ML Model
@st.cache_data
def load_model(model):
    return pickle.load(open(model, 'rb'))

def _show_sparks_gif(img_name: str) -> str:
    """
    Helper method to display gifs

    Args:
        img_name (str): Name of gif as labelled in img folder

    Returns:
        str: URL of gif
    """
    gif = open(os.path.join("img", f"{img_name}.gif"), "rb")
    gif_content = gif.read()
    gif_url = base64.b64encode(gif_content).decode("utf-8")
    gif.close()
    return gif_url

def show_output(position: str, players: list[str], styles: list[str], similar_player_fname: str, similar_player_lname: str, similar_player_id: str):
    """
    Loads playstyle text, similar player and images following the position classification.

    Args:
        position (str): The classified position ("guard", "forward", "center")
        players (list[str]): A list of NBA players who represent a playstyle and whose images will be shown (e.g. "jalen-brunson")
        styles (list[str]): A list of playstyles that will be represented as text (e.g. "Stretch Big")
        similar_player_fname (str):
        similar_player_lname (str):
        similar_player_id (str):
    """
    # Load position classification
    st.image(os.path.join("img", f"{position}-classification.png"))
    st.divider()
    
    # Load most similar player
    st.image(os.path.join("img", "your_nba_comparison.png"))
    
    # Open gifs
    if position == "center":
        gif_url_left = _show_sparks_gif("sparks_red_left")
        gif_url_right = _show_sparks_gif("sparks_red_right")
    elif position == "guard":
        gif_url_left = _show_sparks_gif("sparks_blue_left")
        gif_url_right = _show_sparks_gif("sparks_blue_right")
    else:
        gif_url_left = _show_sparks_gif("sparks_orange_left")
        gif_url_right = _show_sparks_gif("sparks_orange_right")

    # Display gifs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<img src="data:image/gif;base64,{gif_url_left}" alt="cat gif">',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<img src="data:image/gif;base64,{gif_url_right}" alt="cat gif">',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(f"<p style='text-align: center; color: black;'>{similar_player_fname}" + " " + f"{similar_player_lname}</p>", unsafe_allow_html=True)
        st.image(f'player_headshots/{similar_player_id}.png')
        
    st.divider()

    # Display playstyle text and associated player images
    st.image(os.path.join("img", "find-your-playstyle.png"))
    p1, p2, p3, p4, p5, p6 = players[0], players[1], players[2], players[3], players[4], players[5]
    s1, s2, s3, s4, s5, s6 = styles[0], styles[1], styles[2], styles[3], styles[4], styles[5]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<p style='text-align: center; color: black;'>{s1}</p>", unsafe_allow_html=True)
        st.image(f"img/{p1}.png")
        st.text("  ")
        st.markdown(f"<p style='text-align: center; color: black;'>{s2}</p>", unsafe_allow_html=True)
        st.image(f"img/{p2}.png")
    with col2:
        st.markdown(f"<p style='text-align: center; color: black;'>{s3}</p>", unsafe_allow_html=True)
        st.image(f"img/{p3}.png")
        st.text("  ")
        st.markdown(f"<p style='text-align: center; color: black;'>{s4}</p>", unsafe_allow_html=True)
        st.image(f"img/{p4}.png")
    with col3:
        st.markdown(f"<p style='text-align: center; color: black;'>{s5}</p>", unsafe_allow_html=True)
        st.image(f"img/{p5}.png")
        st.text("  ")
        st.markdown(f"<p style='text-align: center; color: black;'>{s6}</p>", unsafe_allow_html=True)
        st.image(f"img/{p6}.png")
        
def collapse_expander():
    """
    Collpases expander container to easily show predicted position for first run of app
    """
    st.session_state['expander'] = False

def main():
    col1, col2 = st.columns(2)
    predicted_pos = -1
    
    # Initialize expander to True
    if 'expander' not in st.session_state:
        st.session_state['expander'] = True
    
    # Define first container
    with col1:
        with st.expander("Physical Dimensions :muscle:", expanded=st.session_state['expander']):
            # Request user input
            st.subheader("What are your physical dimensions?")
            current_year = datetime.now().year
            height = st.number_input("Enter height (m)", min_value=100.00, max_value=250.00, value=None, help="Enter a height between 100-250cm")
            weight = st.number_input("Enter weight (kg)", min_value=30.00, max_value=500.00, value=None, help="Enter a weight between 30-500kg")
            year_start = st.number_input("Enter year you started competitive basketball", min_value=1900, max_value=current_year,
                                         step=1, value=None, help=f"Enter a starting year between: 1900-{current_year}")
            year_end = st.number_input("Enter year you last played competitive basketball", min_value=year_start, max_value=current_year,
                                       value=None, help=f"Enter a ending year between: {year_start}-{current_year}")
            
            # Predict outcome and obtain predicted position
            if st.button("Classify", key = "dimensions-classify", on_click = collapse_expander):
                while (True):
                    try:
                        physical_predictor = load_model(os.path.join("models", "dimensions_rf.sav"))
                        ohe_predictor = load_model(os.path.join("models", "ohe.sav"))
                        input_features = (np.array([[height, weight, year_start, year_end, (weight / (height/100)**2)]]))
                        predicted_pos = ohe_predictor.inverse_transform(physical_predictor.predict(input_features))[0][0]   # obtain prediction using models
                        break 
                    # If model cannot output using input features, slightly adjust height to ensure valid prediction output
                    except ValueError:
                        height += 0.01
                    # If user doesn't provide the necessary details
                    except TypeError:
                        st.warning("Please enter either physical dimensions or player statistics")
                        quit()
                # Obtain most similar player prediction and output
                similar_player_model = load_model(os.path.join("models", "similar_player_dim.pkl"))
                similar_player = similar_player_model.predict_similar_player(height, weight, (weight / (height/100)**2), predicted_pos)
                similar_player_fname, similar_player_lname, similar_player_id = similar_player['fname'], similar_player['lname'], similar_player['playerid']
    
    # Define second container
    with col2:
        with st.expander("Player Statistics :trophy:", expanded=st.session_state['expander']):

            ## Old:
            st.subheader("What are your player statistics?")
            password = st.text_input("Enter Pass", "", key = "stats-input")
            
            model_list = ["LR", "NB"]
            model_choice = st.selectbox("Select ML Model", model_list, key = "stats-model-choice")
            
            if st.button("Classify", key = "stats-classify", on_click = collapse_expander):
                # vect_password = pswd_cv.transform([password]).toarray()
                if model_choice == "LR":
                    prediction = 1
                    # predictor = load_model("models/logit_pswd_model.pkl")
                    # prediction = predictor.predict(vect_password)
                else:
                    prediction = 0
                    # predictor = load_model("models/nv_pswd_model.pkl")
                    # prediction = predictor.predict(vect_password)
    
    # Map index to position (Guard, Forward, Center)
    final_position = get_position(predicted_pos, position_dict)
    
    st.divider()
    
    # Capture case where there is no prediction from model
    if predicted_pos == -1:
        st.warning("Please enter either physical dimensions or player statistics")
    # If valid prediction, output playstyles
    else:
        if final_position == "Guard":
            show_output(position="guard",
                            players=["steph-curry", "jalen-brunson", "chris-paul",
                                      "demar-derozan", "alex-caruso", "jalen-brown"],
                            styles=["3 Point Playmaker", "Back to the basket", "Floor General",
                             "Midrange Shot Creator", "Lockdown Defender", "2 Way Scorer"],
                            similar_player_fname=similar_player_fname,
                            similar_player_lname=similar_player_lname,
                            similar_player_id=similar_player_id,
                            )
        elif final_position == "Forward":
            show_output(position="forward",
                            players=["og-anunoby", "mikal-bridges", "lebron-james",
                                      "giannis-antetokounmpo", "kevin-durant", "kawhi-leonard"],
                            styles=["3&D Wing", "Slashing Playmaker", "Point Forward",
                             "Interior Threat", "Unguardable Unicorn", "3 Level Scorer"],
                            similar_player_fname=similar_player_fname,
                            similar_player_lname=similar_player_lname,
                            similar_player_id=similar_player_id,)
        else:
            show_output(position="center",
                            players=["kristaps-porzingis", "rudy-gobert", "dereck-lively",
                                      "nikola-jokic", "victor-wembanyama", "domantas-sabonis"],
                            styles=["Stretch Big", "Defensive Anchor", "Lob Threat",
                             "Playmaking Big", "Modern Unicorn", "Back To The Basket"],
                            similar_player_fname=similar_player_fname,
                            similar_player_lname=similar_player_lname,
                            similar_player_id=similar_player_id,)
        
    st.markdown("---")

    st.markdown(
        "Source code available at [github.com/danielliu2707/positionn](https://github.com/danielliu2707/positionn)"
    )

    st.markdown(
        "Follow me on [Linkedin](https://www.linkedin.com/in/daniel-liu-80693a20b/)"
    )

if __name__ == "__main__":
    main()