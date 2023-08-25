# Recommendation System IBM 
## Table of Contents
- [Introduction]()
- [File Descriptions]()
- [Installation]()
- [Instructions]()
- [Acknowledgements]()

## Introduction
This project is a component of Udacity's Data Scientist Nanodegree Program, created in partnership with IBM Watson Studio. In the IBM Watson Studio, there is a large collaborative community ecosystem of articles, datasets, notebooks, and other A.I. and ML. assets. Users of the system interact with all of this. Within this scope, we created this recommendation system project to enhance the user experience and connect them with assets. This personalizes the experience for each user.

## File Descriptions

### Folder: data
- **disaster_messages.csv** - Actual messages transmitted during disaster incidents, courtesy of Figure Eight.<br/>
- **disaster_categories.csv** - Categorizations of the aforementioned messages.<br/>
- **process_data.py** - An ETL pipeline script for loading, cleaning, feature extraction, and data storage in an SQLite database.<br/>
- **ETL Pipeline Preparation.ipynb** - A Jupyter Notebook for the development and preparation of the ETL pipeline.<br/>
- **DisasterResponse.db** - The SQLite database containing the cleaned data.

### Folder: models
- **train_classifier.py** - A ML pipeline script that loads cleaned data, trains a model, and stores the trained model in a pickle (.pkl) file for future use.<br/>
- **classifier.pkl** - A pickle file containing the trained model.<br/>
- **ML Pipeline Preparation.ipynb** - A Jupyter Notebook utilized for the development and preparation of the ML pipeline.

## Installation
No additional libraries beyond those included in the Anaconda distribution should be necessary for installation. The code should run without any problems using Python 3.5 or later versions.

## Instructions
1. 

## Acknowledgements:
Thanks to IBM for supplying the dataset that enabled me to train my model.

