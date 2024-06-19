import streamlit as st
import random
import numpy as np
import os
import string
import pickle
from PIL import Image   # For inserting image. You are a guard!!!

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

position_dict = {"Foward": 0, "Center": 2, "Guard": 1, "Please submit either player dimensions or statistics above": -1}

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
            height = st.number_input("Enter height (m)", min_value=100.00, step=0.01)
            weight = st.number_input("Enter weight (kg)", min_value=30.00, step = 0.01)
            
            # Predict outcome and obtain index associated with a players position
            if st.button("Classify", key = "dimensions-classify"):
                physical_predictor = load_model(os.path.join("models", "model.sav"))
                input_features = (np.array([[height, weight, weight / (height/100)**2]]))
                predicted_arr = physical_predictor.predict(input_features)
                
                # [0,0] prediction means a center
                if np.sum(predicted_arr) == 0:
                    predicted_idx = 2
                # [0,1] or [1,0] means Guard or Forward prediction
                else:
                    predicted_idx = int(np.nonzero(predicted_arr)[1])
            
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