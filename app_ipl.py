import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open(r'c:\Users\nauti\Desktop\cpp\python\ipl_win_predictor\pipe.pkl', 'rb'))

st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target')

# Replacing deprecated beta_columns with columns
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets_out = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    remaining_wickets = 10 - wickets_out
    crr = score / overs if overs != 0 else 0  # Avoid division by zero
    rrr = (runs_left * 6) / balls_left if balls_left != 0 else 0  # Avoid division by zero

    # Adding missing column 'wickets_left' (assuming it's the remaining wickets)
    wickets_left = remaining_wickets  # Assuming it's the same as the number of wickets left

    input_df = pd.DataFrame({'batting_team': [batting_team],
                             'bowling_team': [bowling_team],
                             'city': [selected_city],
                             'runs_left': [runs_left],
                             'balls_left': [balls_left],
                             'wickets': [remaining_wickets],
                             'wickets_left': [wickets_left],  # Added missing column
                             'total_runs_x': [target],
                             'crr': [crr],
                             'rrr': [rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(f"{batting_team} - {round(win * 100)}%")
    st.header(f"{bowling_team} - {round(loss * 100)}%")
