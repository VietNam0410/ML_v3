import streamlit as st
from exercises.exercise_1.preprocess import preprocess_data
from exercises.exercise_1.train import train_model
from exercises.exercise_1.demo import show_demo
from exercises.exercise_2.preprocess import preprocess_mnist
from exercises.exercise_2.train import train_mnist
from exercises.exercise_2.demo import show_mnist_demo

st.title("Machine Learning Exercises")

exercise = st.sidebar.selectbox(
    "Chọn một Bài tập",
    ["Exercise 1: Titanic Survival Prediction", "Exercise 2: MNIST Handwritten Digit Recognition"]
)

def display_exercise():
    if exercise == "Exercise 1: Titanic Survival Prediction":
        tab1, tab2, tab3 = st.tabs(["Preprocess Data", "Train Model", "Demo"])
        
        with tab1:
            preprocess_data()
        
        with tab2:
            train_model()
        
        with tab3:
            show_demo()
    # elif exercise == "Exercise 2: MNIST Handwritten Digit Recognition":
    #     tab1, tab2, tab3 = st.tabs(["Preprocess Data", "Train Model", "Demo"])
        
    #     with tab1:
    #         preprocess_mnist()
        
    #     with tab2:
    #         train_mnist()
        
    #     with tab3:
    #         show_mnist_demo()

if exercise in ["Exercise 1: Titanic Survival Prediction", "Exercise 2: MNIST Handwritten Digit Recognition"]:
    display_exercise()