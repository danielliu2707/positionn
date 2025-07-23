<h1 align="center">
  <br>
  <img src="https://github.com/danielliu2707/positionn/blob/main/img/positionn-logo.png">
</h1>

Positionn is an app that predicts your ideal basketball position and finds your NBA twin! 🏀

Simply enter your physical attributes or stats, and with one click, discover your optimal position and the NBA player who matches you best. 🚀  

# Usage

<br>

Try the app <a href="https://positionn.streamlit.app/">here</a> 👈  

### For developers:

Clone repo

```sh
$ git clone https://github.com/danielliu2707/positionn.git
```

Change into directory

```sh
$ cd positionn
```

Create virtual environment

```sh
$ python3 -m venv venv
```

Activate virtual environment

```sh
$ source venv/bin/activate
```

Install requirements

```sh
$ pip install -r requirements.txt
```

Run application

```sh
$ streamlit run app.py
```

# Future roadmap

The following limitations will be prioritised in future updates:

**1. Lack of up-to-date data:** Implement API calls to the NBA API to fetch real-time player statistics and physical attributes from 2010–2025. Use this additional data to build improved models.

**2. Lack of informative outputs:** Display two side-by-side tables: one showcasing the most similar season of a player and the other highlighting their current stats for better comparison.

**3. Extend functionality of application With Computer Vision:** Develop a deep learning model (OpenCV, YOLO etc) that inputs a users full-body photo and predicts their physical dimensions to predict the player most similar to.

**4. Computer vision:** Allow users to upload a photo or video of themselves playing basketball, and use pose estimation models (e.g. OpenPose, MediaPipe) to analyse movement style and suggest an NBA twin using both stats and in-game movement.

**5. Dashboard:** After the application predicts the users position, produce a dashboard, with visualisations such as a histogram showing where they'd be distributed for height, weight etc... compared to the players in the dataset. (Seaborn?)

**6. Need to automate updating data:** Develop a system to automate monthly API calls, ensuring ML models are continuously updated with the latest NBA data.
