# Anime Recommendation System🎌
[![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/-Plotly-3F4F75?logo=plotly&logoColor=white)
[![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![Keras Tuner](https://img.shields.io/badge/-Keras%20Tuner-FF6F00?logo=keras&logoColor=white)](https://keras-team.github.io/keras-tuner/)
[![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/-Kaggle-blue?logo=kaggle)](https://www.kaggle.com/)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?logo=jupyter&logoColor=white)

## Table of Contents

 - [Demo](#demo)
 - [Overview](#overview-)
 - [About the Dataset](#about-the-dataset)
 - [Motivation](#motivation)
 - [Installation](#installation)
 - [Deployement on Streamlit](#deployement-on-streamlit)
 - [Directory Tree](#directory-tree)
 - [Bug / Feature Request](#bug--feature-request)
 - [Future scope of project](#future-scope)
 - [Credits](#credits)

## Demo
- Link: (Not deployed yet)

Note: If the website link provided above is not working, it might mean that the deployment has been stopped or there are technical issues. We apologize for any inconvenience.
- Please consider giving a ⭐ to the repository if you find this app useful.
- A quick preview of my anime recommendation system:

![GIF](resource/loanprediction.gif)

## Overview 🌟📚
Welcome to the Anime Recommendation System! This project aims to provide personalized anime recommendations based on collaborative filtering techniques. 🎉

The application utilizes user-based collaborative filtering to find similar users based on their anime preferences and recommends animes liked by similar users that the target user has not watched yet. Additionally, the system employs item-based collaborative filtering to find similar animes based on their features (e.g., genres, synopsis) and recommends animes similar to the one provided by the user. 🎴📈

The dataset used for training and recommendation includes various anime titles, user ratings, and anime features such as genres and synopses. The model was trained using TensorFlow and Keras to create anime embeddings for both users and animes, facilitating efficient similarity comparisons. 🧠💻

Feel free to explore and enjoy the exciting world of anime recommendations with our innovative system! 💫📺


## Code Walkthrough

Welcome to the Code Walkthrough section of the Anime Recommendation System project! This project is divided into two notebooks, which can be found in the `notebooks` folder of this repository. Let's dive into the two notebooks that make up this awesome project:

#### Notebook 1: `anime-recommendation-1.ipynb` 📒🔍📊
In Notebook 1, we embark on a journey into the world of anime data analysis 🚀📊📈. The main objectives of this notebook are as follows:
- **Understand the dataset**: We take a closer look at the dataset, examining its structure and contents to get familiar with the data.
- **Perform data preprocessing**: We clean and prepare the data for analysis, ensuring that it's in a suitable format for our recommendation models.
- **Interactive Data Visualization**: To gain valuable insights, we utilize the power of [Plotly](https://plotly.com/), a fantastic library that provides us with interactive and engaging visualizations. 📊📈💫

#### Notebook 2: `anime-recommendation-2.ipynb` 📒🔍🤖
In Notebook 2, we take the next step in our journey by training our recommendation model 🚀🤖💡.
- Part 1: Collaborative Filtering 👥🤝

  Here, we delve into collaborative filtering, a popular recommendation technique that suggests animes to users based on the preferences of similar users or similar animes. 🤝📈.

  My key steps were:
  1. **Data Preprocessing**: I loaded the datasets, perform data scaling, and encode user and anime IDs to prepare the data for model training.
  2. **Model Architecture**: To facilitate collaborative filtering, I created a neural network-based model. The model uses embeddings to represent users and animes in a lower-dimensional space, capturing their underlying preferences 🧠🔍.
  3. **Model Training**: Using TensorFlow, I trained the collaborative filtering model to predict user ratings for animes. The optimization process ensures that the model learns to recognize patterns and make accurate recommendations. 📈💡 
  4. **Recommendation Generation**: With the trained model, we can now find similar animes and users 😎.

- Part 2: Content-Based Filtering 📚🔍🎯
  
  The second part of this notebook explores content-based filtering, an alternate recommendation technique. Content-based filtering suggests animes to users based on attributes such as genres and ratings. Here, my key steps were:
  1. **TF-IDF Vectorization**: I created a TF-IDF matrix for anime genres to quantify the importance of genres in each anime's description.
  2. **Cosine Similarity**: By computing cosine similarity between animes based on their genre descriptions, we can determine their similarity.
  3. **Content-Based Recommendation**: Leveraging the computed similarity scores and ratings, we now can recommend animes that are similar to a given anime, considering their genre and score.

  We've got an exciting mix of collaborative and content-based filtering models, ensuring we can deliver diverse and accurate anime recommendations tailored to the preferences of each user. 🤗

  Happy anime recommending! 🎊

#### `NOTE`:-
Due to the large size of the model file i.e `myanimeweights.h5`, it cannot be directly hosted on GitHub. However, you can still access and download the files from my [Google Drive](https://drive.google.com/file/d/1dypdLwProMGxy7h8hi49a-kHiu_nFz1U/view?usp=sharing).

## About the Dataset




## Motivation

During my second year of college, I discovered my passion for Data Science. As I delved deeper into the subject, I was fascinated by the mathematics behind supervised and unsupervised machine learning models. However, I realized that it's not enough to just understand the theory - it's important to apply this knowledge to real-world problems to make a meaningful impact.
That's why I decided to build this loan status predictor app. By leveraging the power of artificial neural networks, this app can accurately predict whether a loan will be fully paid or charged off. Through this project, I hope to not only demonstrate my skills in Data Science but also make a positive contribution to the financial industry by improving risk assessment for lenders.

## Installation

This project is written in Python 3.10.10. If you don't have Python installed, you can download it from the [official website](https://www.python.org/downloads/). If you have an older version of Python, you can upgrade it using the pip package manager, which should be already installed if you have Python 2 >=2.7.9 or Python 3 >=3.4 on your system.
To install the required packages and libraries, you can use pip and the provided requirements.txt file. First, clone this repository to your local machine using the following command:
```
git clone https://github.com/Sajid030/Lending-Club-Loan-Prediction.git
```
Once you have cloned the repository, navigate to the project directory and run the following command in your terminal or command prompt:
```bash
pip install -r requirements.txt
```
This will install all the necessary packages and libraries needed to run the project.

## Deployement on Streamlit
1. Create an account on Streamlit Sharing.
2. Fork this repository to your GitHub account.
3. Log in to Streamlit Sharing and create a new app.
4. Connect your GitHub account to Streamlit Sharing and select this repository.
5. Set the following configuration variables in the Streamlit Sharing dashboard:
```
[server]
headless = true
port = $PORT
enableCORS = false
```
6. Click on "Deploy app" to deploy the app on Streamlit Sharing.

## Directory Tree

```
├── model
│   ├── dataset.pkl
│   ├── my_model.h5
├── resource 
│   ├── loanprediction.gif
├── app.py
├── LICENSE.md
├── loan_prediction.ipynb
├── README.md
├── requirements.txt
```

## Bug / Feature Request

If you encounter any bugs or issues with the loan status predictor app, please let me know by opening an issue on my [GitHub repository](https://github.com/Sajid030/Lending-Club-Loan-Prediction/issues). Be sure to include the details of your query and the expected results. Your feedback is valuable in helping me improve the app for all users. Thank you for your support!

## Future Scope

- Improving the model performance by trying different machine learning algorithms or hyperparameter tuning.
- Adding more features to the dataset, which could potentially improve the accuracy of the model.
- Deploying the model on a cloud platform like AWS, GCP or Azure for more scalable and reliable use.
- Integrating the model with other financial data sources to make more accurate predictions and provide better insights.
- Using natural language processing techniques to analyze borrower comments or reviews to identify any potential risks or fraud.

## Credits
- Dataset link : https://www.kaggle.com/datasets/janiobachmann/lending-club-first-dataset
- Dataset features details :https://github.com/dosei1/Lending-Club-Loan-Data/blob/master/LCDataDictionary.csv
