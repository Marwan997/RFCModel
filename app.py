import streamlit as st
import joblib
import numpy as np

#load model
model = joblib.load('tennis_rfc_model.pkl')

def main():
    st.title('Model Deployment')

    #input components for each feature
    points_diff = st.text_input('Points Difference')
    age_diff = st.text_input('Age Difference')
    rs_diff = st.text_input('RS Difference')
    rs_win_diff = st.text_input('RS Win Difference')
    rs_win_p_diff = st.text_input('RS Win Percentage Difference')
    yearly_win_diff = st.text_input('Yearly Win Difference')
    p1_matches_played = st.text_input('Player 1 Matches Played')
    p1_peak_elo = st.text_input('Player 1 Peak Elo')
    p2_matches_played = st.text_input('Player 2 Matches Played')
    p2_peak_elo = st.text_input('Player 2 Peak Elo')
    rank_diff = st.text_input('Rank Difference')

    #validate and preprocess input data
    if (
        points_diff
        and age_diff
        and rs_diff
        and rs_win_diff
        and rs_win_p_diff
        and yearly_win_diff
        and p1_matches_played
        and p1_peak_elo
        and p2_matches_played
        and p2_peak_elo
        and rank_diff
    ):
        try:
            #convert input data to float and create feature array
            input_features = np.array(
                [
                    float(points_diff),
                    float(age_diff),
                    float(rs_diff),
                    float(rs_win_diff),
                    float(rs_win_p_diff),
                    float(yearly_win_diff),
                    float(p1_matches_played),
                    float(p1_peak_elo),
                    float(p2_matches_played),
                    float(p2_peak_elo),
                    float(rank_diff),
                ]
            ).reshape(1, -1)

            #make predictions using the loaded model
            prediction = model.predict(input_features)

            #results
            st.write('Prediction:', prediction[0])
        except ValueError:
            st.error('Please enter valid numeric values for all input features.')
    else:
        st.info('Please enter values for all input features.')

if __name__ == '__main__':
    main()