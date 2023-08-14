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
- [Code Walkthrough](#code-walkthrough)
- [About the Dataset](#about-the-dataset)
- [Motivation](#motivation)
- [Acknowledgments](#acknowledgments)
- [Installation](#installation)
- [Directory Tree](#directory-tree)
- [Bug / Feature Request](#bug--feature-request)
- [Future scope of project](#future-scope)

## Demo

- Link: (Not deployed yet)

Note: If the website link provided above is not working, it might mean that the deployment has been stopped or there are technical issues. We apologize for any inconvenience.

- Please consider giving a ⭐ to the repository if you find this app useful.
- A quick preview of my anime recommendation system:

https://github.com/Sajid030/anime-recommendation-system/assets/126476034/756fafed-7caf-4811-af46-8a8888e6f14b

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

The dataset used in the Anime Recommendation System project offers a wealth of valuable information, encompassing anime characteristics, ratings, popularity, viewership, user behavior, and preferences. It serves as a comprehensive resource for conducting diverse analyses, such as identifying top-rated anime, exploring popular genres, and gaining insights into viewer trends. With this dataset, personalized recommendation systems can be developed, user behavior can be analyzed, and clustering can be employed to understand anime trends and user preferences. Additionally, the dataset enables examination of user interactions, ratings, and engagement with anime, providing essential inputs for collaborative filtering and similarity analysis. Overall, this dataset is instrumental in building an effective recommendation system and deepening our understanding of anime trends and user preferences on the platform. 📈🎌📊

This dataset was built by me. For a detailed overview of the dataset creation process, you can visit my [GitHub repository](https://github.com/Sajid030/anime_dataset_generator), where I have explained the procedure I followed to generate the complete dataset, including the tools and techniques used to create the final dataset.

You can find my complete dataset on [kaggle](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset). If you find this dataset helpful, please consider giving it an upvote to show your appreciation.😊

## Motivation

As an anime weeb, I have always been captivated by the fascinating world of anime. The colorful characters, gripping storylines, and imaginative worlds never fail to spark my imagination and evoke a sense of wonder. It didn't take long for me to realize that the joy I experienced while watching animes could be elevated to another level by bringing the magic of anime recommendations into real life. The idea of creating an Anime Recommendation System took root in my mind, and I embarked on this exciting journey to develop a platform that could intelligently suggest animes based on individual preferences. This project allowed me to combine my passion for anime with the fascinating realm of data science and machine learning, enabling me to create a personalized anime discovery experience for myself and fellow anime enthusiasts. With every line of code written and every model trained, I found myself one step closer to bridging the gap between my love for anime and the power of technology to make anime-watching experiences even more enjoyable. Join me as I unravel the world of anime recommendations and witness the magic of data-driven suggestions that open up new horizons in the anime universe! 🌟🎬🚀

## Acknowledgments

A big shoutout to my friend [Aditya Singh Rawat](https://github.com/unreal0901) for their invaluable contribution to this project. Their design skills and collaboration were essential in bringing the Anime Recommendation App to life. 🙌

## Installation

This project is written in Python 3.11.4. If you don't have Python installed, you can download it from the [official website](https://www.python.org/downloads/). If you have an older version of Python, you can upgrade it using the pip package manager, which should be already installed if you have Python 2 >=2.7.9 or Python 3 >=3.4 on your system.
To install the required packages and libraries, you can use pip and the provided requirements.txt file. First, clone this repository to your local machine using the following command:

```
https://github.com/Sajid030/anime-recommendation-system.git
```

Once you have cloned the repository, navigate to the project directory and run the following command in your terminal or command prompt:

```bash
pip install -r requirements.txt
```

This will install all the necessary packages and libraries needed to run the project.

If you prefer, you can also create a virtual environment to manage the project dependencies separately. This helps in isolating the project's environment from the system-wide Python installation.

## Directory Tree

```
|   .gitignore
|   app.py
|   LICENSE.md
|   requirements.txt
|
+---model
|       anime-dataset-2023.pkl
|       anime_encoder.pkl
|       myanimeweights.h5
|       users-score.csv
|       user_encoder.pkl
|
+---notebooks
|       anime-recommendation-1.ipynb
|       anime-recommendation-2.ipynb
|
+---static
|       style.css
|
\---templates
        index.html
        recommendations.html
```

## Bug / Feature Request

If you encounter any bugs or issues with the anime recommendation app, please let me know by opening an issue on my [GitHub repository](https://github.com/Sajid030/anime-recommendation-system/issues). Be sure to include the details of your query and the expected results. Your feedback is valuable in helping me improve the app for all users. Thank you for your support!

## Future Scope

1. **Implement Hybrid Recommendation System**: Combine collaborative filtering and content-based filtering techniques to create a hybrid recommendation system.
2. **Include User Feedback and Reviews**: Incorporate user feedback and reviews into the recommendation system to improve the accuracy of recommendations.
3. **Explore Deep Learning Models**: Experiment with advanced deep learning models, such as RNNs and transformer-based architectures, to enhance recommendation performance.
4. **Real-Time Recommendation Updates**: Implement a real-time recommendation system that continuously updates suggestions based on users' interactions.
5. **Integrate External Data Sources**: Consider integrating external data sources, such as user demographics and anime-related news, to personalize recommendations.
6. **Anime Sentiment Analysis**: Perform sentiment analysis on anime reviews to gauge audience sentiments towards specific animes.
7. **User Clustering**: Cluster users based on preferences to provide better personalized recommendations and targeted marketing strategies.
8. **Interactive Web Interface**: Develop a user-friendly web interface for exploring recommendations and detailed anime information.
9. **Social Media Integration**: Allow users to share favorite animes and recommendations on social media platforms.
10. **Anime Popularity Trend Analysis**: Conduct time series analysis to identify trends in anime popularity over different seasons and years.
11. **Personalized Watchlists**: Create personalized watchlists for users, curating a list of animes based on their preferences.
12. **Sentiment-Based Filtering**: Implement sentiment-based filtering for recommending animes based on users' emotions.
