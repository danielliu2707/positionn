import streamlit as st
import random
import os
import string

st.set_page_config(
    page_title = "Physical Dimensions",
    page_icon = "🏀",
    initial_sidebar_state="collapsed"
)

# Hide top multicolor header
hide_decoration_bar_style = '''<style>header {visibility: hidden;}</style>'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)



## st.image()

st.markdown("Determine what NBA position would most suit a player based on Physical Dimensions and/or Statistics.")

st.sidebar.write("Enter physical dimensions to see what position is most suitable")


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

password_labels = {"weak": 0, "average":1, "strong": 2}

# def load_model(model_file):
#     loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
#     return loaded_model
        
def password_generator(size: int):
    characters = string.digits + string.punctuation + string.ascii_letters
    generated_password = "".join(random.choice(characters) for x in range(size))
    return generated_password


def main_physical():
    
    # st.image()
    
    
    with st.expander("Physical Dimensions"):
        
        ## Old:
        st.subheader("What are your player dimensions?")
        password = st.text_input("Enter Password", "", key = "dimensions-input")
        
        model_list = ["LR", "NB"]
        model_choice = st.selectbox("Select ML Model", model_list, key = "dimensions-model-choice")
        
        if st.button("Classify", key = "dimensions-classify"):
            # vect_password = pswd_cv.transform([password]).toarray()
            if model_choice == "LR":
                prediction = 1
                # predictor = load_model("models/logit_pswd_model.pkl")
                # prediction = predictor.predict(vect_password)
            else:
                prediction = 0
                # predictor = load_model("models/nv_pswd_model.pkl")
                # prediction = predictor.predict(vect_password)
            final_result = get_key(prediction, password_labels)
            st.info(final_result)
        ##
    
    with st.expander("Player Statistics"):

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
            final_result = get_key(prediction, password_labels)
            st.info(final_result)
        ##
            
    st.markdown("---")

    st.markdown(
        "Source code available at [github.com/danielliu2707/positionn](https://github.com/danielliu2707/positionn)"
    )

    st.markdown(
        "Follow me on [Linkedin](https://www.linkedin.com/in/daniel-liu-80693a20b/)"
    )


if __name__ == "__main__":
    main_physical()