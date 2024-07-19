import streamlit as st
import random
import numpy as np
import os
import string
import pickle
from PIL import Image   # For inserting image. You are a guard!!!
from datetime import datetime

st.set_page_config(
    page_title = "positionn",
    page_icon = ":basketball:",
    initial_sidebar_state="collapsed"
)

# Hide top multicolor header
hide_decoration_bar_style = '''<style>header {visibility: hidden;}</style>'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

# Define logo 
st.image(os.path.join("img", "positionn-logo.png"))
st.markdown("Determine what NBA position best suits a player based on their Physical Dimensions or Statistics.")

def get_position(predicted_idx, my_dict):
    for key, value in my_dict.items():
        if value == predicted_idx:
            return key

position_dict = {"Foward": "F", "Center": "C", "Guard": "G", "Please submit either player dimensions or statistics above": -1}

# Loading in ML Model
@st.cache_data
def load_model(model):
    return pickle.load(open(model, 'rb'))

def main_physical():
    col1, col2 = st.columns(2)
    predicted_idx = -1
    
    with col1:
        with st.expander("Physical Dimensions :muscle:", expanded=True):
            st.subheader("What are your physical dimensions?")
            # Request user input and convert into matrix
            current_year = datetime.now().year
            height = st.number_input("Enter height (m)", min_value=100.00, max_value=250.00, value=None, help="Enter a height between 100-250cm")
            weight = st.number_input("Enter weight (kg)", min_value=30.00, max_value=500.00, value=None, help="Enter a weight between 30-500kg")
            year_start = st.number_input("Enter the year you started competitive basketball", min_value=1900, max_value=current_year, step=1, value=None, help=f"Enter a starting year between: 1900-{current_year}")
            year_end = st.number_input("Enter the year you stopped competitive basketball", min_value=year_start, max_value=current_year, value=None, help=f"Enter a ending year between: {year_start}-{current_year}")
            # Predict outcome and obtain index associated with a players position
            if st.button("Classify", key = "dimensions-classify"):
                while (True):
                    try:
                        physical_predictor = load_model(os.path.join("models", "dimensions_rf.sav"))
                        ohe_predictor = load_model(os.path.join("models", "ohe.sav"))
                        input_features = (np.array([[height, weight, year_start, year_end, (weight / (height/100)**2)]]))
                        predicted_idx = ohe_predictor.inverse_transform(physical_predictor.predict(input_features))[0][0]
                        break
                    except ValueError:
                        height += 0.01
    with col2:
        with st.expander("Player Statistics :trophy:", expanded=True):

            ## Old:
            st.subheader("What are your player statistics?")
            password = st.text_input("Enter Pass", "", key = "stats-input")
            
            model_list = ["LR", "NB"]
            model_choice = st.selectbox("Select ML Model", model_list, key = "stats-model-choice")
            
            if st.button("Classify", key = "stats-classify"):
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
    final_result = get_position(predicted_idx, position_dict)
    
    if predicted_idx == -1:
        st.warning(final_result)
    else:
        ## TODO: Modify the following code to output what I desire. Example: Position, Some Details of the position such as avg dimensions for dimension
        ## and average statistics for stats. Also, all stars at this level. Basically produce a dashboard for this statistic... Note: Can do this later...
        ## Define function for achieving this
        st.info(final_result)
    
                
    st.markdown("---")

    st.markdown(
        "Source code available at [github.com/danielliu2707/positionn](https://github.com/danielliu2707/positionn)"
    )

    st.markdown(
        "Follow me on [Linkedin](https://www.linkedin.com/in/daniel-liu-80693a20b/)"
    )

if __name__ == "__main__":
    main_physical()