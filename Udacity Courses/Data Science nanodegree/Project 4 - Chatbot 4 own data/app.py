# Standard library imports
import os
import sys
import time
import pickle
from pprint import pprint
from dotenv import find_dotenv, load_dotenv

# Third-party library imports
import gradio as gr
import openai

# Langchain specific imports
from langchain.agents import AgentExecutor, tool
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.vectorstores.base import VectorStoreRetriever


# Load environment variables from .env file, exit on error
if not load_dotenv(find_dotenv()):    
    sys.exit("Error: .env file not found.")

# Try to assign the OpenAI API key from the environment variable to openai.api_key.
try:     
    openai.api_key = os.environ['OPENAI_API_KEY']

# If the OPENAI_API_KEY environment variable is not set, exit the program with an error message.
except KeyError:     
    sys.exit("Error: OPENAI_API_KEY environment variable not set.")


def load_retriever(filename: str = "vectorstore.pkl") -> 'VectorStoreRetriever':
    """
    Loads a VectorStoreRetriever with a vector store from a specified file.

    Parameters:
    - filename (str): The name of the file containing the vector store.

    Returns:
    - VectorStoreRetriever: The initialized VectorStoreRetriever object.
    """
    try:
        # Attempt to open and read the specified file in binary read mode.
        with open(filename, "rb") as f:

            # Load the vector store data using pickle.
            vectorstore = pickle.load(f)

    except (FileNotFoundError, pickle.UnpicklingError) as e:

        # If file is not found or unpickling fails, raise a runtime error with a descriptive message.
        raise RuntimeError(f"Failed to load vector store: {e}")

     # Initialize a VectorStoreRetriever object with the loaded vector store data and specific search arguments.
    retriever = VectorStoreRetriever(vectorstore=vectorstore, search_kwargs={"k": 2})
    
    # Return the initialized VectorStoreRetriever object.
    return retriever


# This is a decorator which associates some additional behavior or attributes to the function below
@tool
def tool(query):
    "Searches and returns documents regarding the oil company Shell"
    docs = retriever.get_relevant_documents(query)
    return docs

######### Configuration and initialization #####

# This is needed for both the memory and the prompt
memory_key = "history"

# load retriever
retriever = load_retriever()

# define list of tools
tools = [tool]

# Add memory to make sure it remembers previous interactions
memory = ConversationBufferMemory(return_messages=True, memory_key=memory_key)

# Prompt template that defines general instructions as how the agent needs to behave
system_message = SystemMessage(
        content=(
            "Do your best to answer the questions. "
            "Feel free to use any tools available to look up "
            "relevant information, only if neccessary."        
        )
)

# Define prompt and add conversation history
prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
    )

# Instantiate the agent
llm = ChatOpenAI(temperature = 0)
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)


# Define the agent executor linking all elements together
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)


######### Build the gradio app #####

def build_and_launch_gradio_interface():
    """
    Build and launch a Gradio interface for a chatbot application.

    This function constructs a Gradio interface that includes a chatbot,
    a text input box for user input, and two buttons for clearing chat history
    and memory. It handles user interaction by sending user messages to an
    agent executor and a document retriever, which processes the message
    and fetches relevant documents respectively.

    The bot's response, along with the formatted references from the documents,
    is displayed in the chatbox. The user can clear the chat history 
    by clicking the respective button.

    This function should be called in the script's main execution block to
    ensure the Gradio interface is built and launched when the script is run.

    """

    # Clear any previous memory
    memory.clear()

    # Initiate Gradio Blocks interface
    with gr.Blocks() as demo:

        # Create a Chatbot UI element
        chatbot = gr.Chatbot()

        # Create a Textbox UI element for user input
        msg = gr.Textbox()

        # Create a Button UI element to clear chat history
        clear_history = gr.Button("Clear chat history")

        # Initialize an empty chat history
        chat_history = []
        
        def user(user_message, history):
            """
            Handle user message and update the chat history.

            This function takes a user message and the current chat history, 
            appends the user message to the history, and returns an empty 
            bot message alongside the updated chat history.

            Parameters:
                user_message (str): The message inputted by the user in the Gradio interface.
                history (list): The current chat history, formatted as a list of 
                               [user_message, bot_message] pairs.

            Returns:
                tuple: A tuple containing an empty string representing the bot's 
                       message and the updated chat history.
            """
            return "", history + [[user_message, None]]
        
        def bot(history): 
            """
            Process the latest user message, fetch relevant documents, and compose a bot response.

            This function is triggered after a user submits a message. It executes an agent with
            the latest user message, retrieves documents relevant to the message, formats the 
            document references, and composes a bot response. The response is generated character-by-character
            to simulate a typing effect in the chat interface.

            Parameters:
                history (list): The current chat history, formatted as a list of 
                               [user_message, bot_message] pairs.

            Yields:
                list: The updated chat history with the bot's response appended to the latest
                      entry, yielding at each character to update the chat interface.      
            """

            # Execute agent with the latest user message and get a response
            result = agent_executor({"input": history[-1][0]})

            # Extract bot message from the result            
            bot_message = result['output']

            # Retrieve documents relevant to the latest user message
            extracted_results = retriever.get_relevant_documents(history[-1][0])

            # Format retrieved documents' source and page number
            formatted_results = ["source: {} page {}".format(doc.metadata["source"], doc.metadata["page"]) for doc in extracted_results]

            # Combine bot message and formatted document references into a full message
            full_message = bot_message + '\n\n' + '\n\n'.join(formatted_results)
            
            # Initialize an empty bot response slot in the latest chat history entry
            history[-1][1] = ""

            # Character-by-character mimicking a typing effect
            for character in full_message:
                history[-1][1] += character

                 # Delay to simulate typing
                time.sleep(0.02) 

                # Update the chatbot UI with the current state of the message
                yield history  

        # Setup user message submission, triggering the `bot` function upon submission
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)

        # Setup the clear chat history button, with a no-op lambda function as the click handler
        clear_history.click(lambda: None, None, chatbot, queue=False)     
       
    # Queue the demo interface for launch
    demo.queue()

    # Launch the Gradio interface
    demo.launch()


# Run the Gradio Interface
if __name__ == "__main__":
    build_and_launch_gradio_interface()