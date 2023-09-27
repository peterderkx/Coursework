# Chatbot 4 own data
## Table of Contents
- [Introduction](https://github.com/peterderkx/Coursework/tree/main/Udacity%20Courses/Data%20Science%20nanodegree/Project%204%20-%20Chatbot%204%20own%20data#introduction)
- [File Descriptions](https://github.com/peterderkx/Coursework/tree/main/Udacity%20Courses/Data%20Science%20nanodegree/Project%204%20-%20Chatbot%204%20own%20data#file-descriptions)
- [Installation](https://github.com/peterderkx/Coursework/tree/main/Udacity%20Courses/Data%20Science%20nanodegree/Project%204%20-%20Chatbot%204%20own%20data#installation)
- [Instructions](https://github.com/peterderkx/Coursework/tree/main/Udacity%20Courses/Data%20Science%20nanodegree/Project%204%20-%20Chatbot%204%20own%20data#instructions)
- [Acknowledgements](https://github.com/peterderkx/Coursework/tree/main/Udacity%20Courses/Data%20Science%20nanodegree/Project%204%20-%20Chatbot%204%20own%20data#acknowledgements)
- [Screenshots](https://github.com/peterderkx/Coursework/tree/main/Udacity%20Courses/Data%20Science%20nanodegree/Project%204%20-%20Chatbot%204%20own%20data#screenshots)

## Introduction
This repository is created to create a chatbot on your own data and a user interface built with [Gradio](https://www.gradio.app/).
The chatbot will also return the reference docs and pages.

You would need a openAI key to interact with their API engine, please visit [openAI](https://openai.com/).
Once you have a key, save it in the .env file

I stored my files in the docs folder. The files are (publicly published) Shell annual and sustainability reports.
To load, embed and save mebddings, a seperate file is created, please see below. As such the costly part of creating embddings is already done. The load of the vectorstore is done in the app.py

In essence the following process is followed in this file:
- load data, split data into chunks, create (vector) embeddings of chunks with FAISS, save vectorstore

Then to run the app the following process is followed in the app.py file:
- load vectorstore, create template for prompt, create agent for extraction which has a conversation buffer for memory, gradio UI that keeps track of conversation and streams responses

A demo video is uploaded as well to show the chatbot in action.

## File Descriptions

### Folder: docs
- Folder: Contains the own data I want to access with the Chatbot. 

### files
- **load_data_save_embed.py** - python file for loading data, generate embeddings and save data into a vectorstore for retrieval
- **app.py** - python file that upon running will open the chatbot that allows you to interact with your data
- **vectorsore.pkl** - pickle file with a vecorstore of saved embeddings that can be used for retrieval after loading
- **demo video.mkv** - video file that demonstrates the use of the chatbot
- **requirements.txt** - list of required packages to install
- **.env** - file that contains the openAI key. Use your own key as there are costs involved upon usage
- **qa for testing.xlxs** - list of questions to test and validate the chatbot

## Installation
No specific installations are required, all libraries are imported, you might need to pip install them depending on whether you have already installed them before.

## Instructions
1. Run the following commands in the project's root directory if you need to create embeddings of your data. In order to save time they have already been created and saved for you, but if you want to add new data files, then you can do so.

    - To load and save embeddings:
        `load_data_save_embed.py`

2. Run the following command in the app's directory to run your web app.
    `python app.py`

3. Go to http://127.0.0.1:7860 or another link that will be provided in the terminal (Gradio standard)

## Acknowledgements:
Thanks to LangChain, Gradio and OpenAI for providing the necessary guidance on how to create such a bot

## Screenshots
- Screenshot of the chatbot. Please check the video as well for a better demo<br/>
![image](https://github.com/peterderkx/Coursework/blob/main/Udacity%20Courses/Data%20Science%20nanodegree/Project%204%20-%20Chatbot%204%20own%20data/screenshot%20chatbot%204%20own%20data.png) <br/>
