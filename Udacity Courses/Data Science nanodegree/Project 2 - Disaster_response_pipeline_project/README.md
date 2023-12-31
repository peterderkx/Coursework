# Disaster Response Pipeline Project
## Table of Contents
- [Introduction](https://github.com/peterderkx/Coursework/tree/main/Udacity%20Courses/Data%20Science%20nanodegree/Project%202%20-%20Disaster_response_pipeline_project#introduction)
- [File Descriptions](https://github.com/peterderkx/Coursework/tree/main/Udacity%20Courses/Data%20Science%20nanodegree/Project%202%20-%20Disaster_response_pipeline_project#file-descriptions)
- [Installation](https://github.com/peterderkx/Coursework/tree/main/Udacity%20Courses/Data%20Science%20nanodegree/Project%202%20-%20Disaster_response_pipeline_project#installation)
- [Instructions](https://github.com/peterderkx/Coursework/tree/main/Udacity%20Courses/Data%20Science%20nanodegree/Project%202%20-%20Disaster_response_pipeline_project#instructions)
- [Acknowledgements](https://github.com/peterderkx/Coursework/tree/main/Udacity%20Courses/Data%20Science%20nanodegree/Project%202%20-%20Disaster_response_pipeline_project#acknowledgements)
- [Screenshots](https://github.com/peterderkx/Coursework/tree/main/Udacity%20Courses/Data%20Science%20nanodegree/Project%202%20-%20Disaster_response_pipeline_project#screenshots)

## Introduction
This project is a component of Udacity's Data Scientist Nanodegree Program, created in partnership with [Figure Eight](https://www.figure-eight.com/). It involves the development of a disaster response model that leverages pre-labeled disaster messages to categorize real-time messages during a disaster event, ensuring that they are directed to the appropriate response agency. Additionally, this project features a web application through which disaster response personnel can input incoming messages and receive classification results.

For the 'rescue center' under consideration, it is likely that recall is prioritized to ensure no one in need is left behind. However, high precision is equally important to avoid wasting resources on false alarms. The relative importance of these metrics can shift depending on the specific circumstances and available resources. Ideally, striking a balance between recall and precision will allow for efficient resource allocation while still addressing genuine emergencies. The data provided shows marked improvements in recall scores for crucial rescue service categories, indicating better detection of general aid needs. There are also significant advancements in the identification of critical essentials. In summary, the new system shows enhanced effectiveness in recognizing rescue needs.

## File Descriptions

### Folder: app
- **run.py** - A Python script used to initiate the web application.<br/>
- Folder: templates - Contains the necessary web dependency files (go.html & master.html) for operating the web application.

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

### Folder: screenshots
- screenshots of the application and training, please see remarks below as well
## Installation
No additional libraries beyond those included in the Anaconda distribution should be necessary for installation. The code should run without any problems using Python 3.5 or later versions.

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgements:
Thanks to Figure Eight for supplying the dataset that enabled me to train my model.

## Screenshots
- The main page displays an overview of the Training Dataset and the Distribution of Message Categories.<br/>
![image](https://github.com/peterderkx/Coursework/blob/main/Udacity%20Courses/Data%20Science%20nanodegree/Project%202%20-%20Disaster_response_pipeline_project/screenshots/20230814%20-%20app%20screenshot%201.PNG) <br/>
- To classify a message, enter the message and click on the 'Classify Message' button. <br/>
![image](https://github.com/peterderkx/Coursework/blob/main/Udacity%20Courses/Data%20Science%20nanodegree/Project%202%20-%20Disaster_response_pipeline_project/screenshots/20230814%20-%20app%20screenshot%202.PNG) <br/>
- After clicking 'Classify Message', the category or categories to which the message belongs will be highlighted in green. <br/>
![image](https://github.com/peterderkx/Coursework/blob/main/Udacity%20Courses/Data%20Science%20nanodegree/Project%202%20-%20Disaster_response_pipeline_project/screenshots/20230814%20-%20app%20screenshot%203.PNG) <br/>
- Execute the process_data.py script to run the ETL pipeline. <br/>
![image](https://github.com/peterderkx/Coursework/blob/main/Udacity%20Courses/Data%20Science%20nanodegree/Project%202%20-%20Disaster_response_pipeline_project/screenshots/20230814%20-%20ETL%20screenshot.PNG) <br/>
- Execute the train_classifier.py script to run the ML pipeline. <br/>
![image](https://github.com/peterderkx/Coursework/blob/main/Udacity%20Courses/Data%20Science%20nanodegree/Project%202%20-%20Disaster_response_pipeline_project/screenshots/20230814%20-%20classifier%20screenshot%201.PNG) <br/>
![image](https://github.com/peterderkx/Coursework/blob/main/Udacity%20Courses/Data%20Science%20nanodegree/Project%202%20-%20Disaster_response_pipeline_project/screenshots/20230814%20-%20classifier%20screenshot%202.PNG)
- In the app's directory, run the run.py script to launch the web application. <br/>
![image](https://github.com/peterderkx/Coursework/blob/main/Udacity%20Courses/Data%20Science%20nanodegree/Project%202%20-%20Disaster_response_pipeline_project/screenshots/20230814%20-%20app%20screenshot%204.PNG) <br/>