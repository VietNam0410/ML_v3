import streamlit as st
from exercises.exercise_1.preprocess import preprocess_data
from exercises.exercise_1.train import train_model
from exercises.exercise_1.demo import show_demo

st.title("Machine Learning Exercises")

exercise = st.sidebar.selectbox("Choose an Exercise", ["Exercise 1: Titanic Survival Prediction"])

def display_exercise():
    tab1, tab2, tab3 = st.tabs(["Preprocess Data", "Train Model", "Demo"])
    
    with tab1:
        preprocess_data()
    
    with tab2:
        train_model()
    
    with tab3:
        show_demo()

if exercise == "Exercise 1: Titanic Survival Prediction":
    display_exercise()