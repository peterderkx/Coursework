import logging
import os
import pickle
import sys
from dotenv import find_dotenv, load_dotenv
import openai
from pprint import pprint
from langchain.document_loaders import DirectoryLoader, PDFPlumberLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Setup logging
logging.basicConfig(level=logging.INFO)

def setup_openai_api_key():
    """
    Set up the OpenAI API key from the environment variable.
    
    Raises:
        SystemExit: If the OPENAI_API_KEY environment variable is not set.
    """
    try:
        openai.api_key = os.environ['OPENAI_API_KEY']
    except KeyError:
        logging.error("OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

def load_documents_from_directory(directory, file_extension, loader_class):
    """
    Load documents from a specified directory with a specified file extension.
    
    Parameters:
    - directory (str): The path to the directory containing the documents.
    - file_extension (str): The file extension to search for.
    - loader_class (type): The class to use for loading the documents.
    
    Returns:
        list: A list of loaded documents.
    """
    loader = DirectoryLoader(directory, glob=f"./*{file_extension}", loader_cls=loader_class)
    return loader.load()

def main():
    """
    Main function to execute the script. It performs the following steps:
    1. Load environment variables from the .env file.
    2. Set up the OpenAI API key.
    3. Load documents from the specified directory.
    4. Split the loaded documents in chuncks.
    5. Initialize OpenAI embeddings.
    6. Create a FAISS vector store from the documents and embeddings.
    7. Save the FAISS vector store to a file.
    """
        
    # 'Load environment variables from .env file'
    dotenv_file_path = find_dotenv()
    _ = load_dotenv(dotenv_file_path)

    # Set up the OpenAI API key.
    setup_openai_api_key()

    # Configuration for document loaders
    CONFIG = {
        'directory': 'docs',
        'loaders': [
            {'extension': '.pdf', 'loader_class': PDFPlumberLoader},
            {'extension': '.txt', 'loader_class': TextLoader}
        ]
    }

    # Load documents using the specified loaders
    pprint('Loading...')
    documents = []
    for loader_config in CONFIG['loaders']:
        documents.extend(load_documents_from_directory(
            CONFIG['directory'],
            loader_config['extension'],
            loader_config['loader_class']
        ))

    # Define chuck size and chunck overlap size
    chunk_size = 2000
    chunk_overlap = 200

    # Initialize character text splitter with a chunk size and chunk overlap
    character_text_splitter = CharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    
    # Split documents into chunks using the character text splitter
    pprint('Splitt docs into chuncks...')
    split_documents = character_text_splitter.split_documents(documents)

    # Initialize OpenAI embeddings
    pprint('Create embeddings...')
    embeddings = OpenAIEmbeddings()     
    
    # Initialize FAISS vector store from documents and embeddings
    faiss_vector_store = FAISS.from_documents(split_documents, embeddings)

    # Save FAISS vector store to a file
    pprint('Save embeddings...')
    try:
        with open("vectorstore.pkl", "wb") as f:
            pickle.dump(faiss_vector_store, f)
    except IOError as e:
        logging.error(f"Unable to write to file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

