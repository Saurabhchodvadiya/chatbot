# Importing the libraries (you won't need all of them right now but you will need them later)

from flask import Flask, request, jsonify
from flask import Flask, render_template, request, url_for
# from llama_index import SimpleDirectoryReader, GPTListIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os
import time
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas
import openai
import numpy as np
import glob
import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

import os
import glob
import pandas as pd
import openai
# from langchain import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-proj-Ng6RQQY78yRlutrkP6YnT3BlbkFJ2jFHrWMayruEjgQCAtcO"
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize the OpenAI model
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# Get the list of PDF files
gfiles = glob.glob(r"C:\Users\parth\Downloads\forexbot2\forexbot2\forex_trading_tutorial.pdf")

# Iterate through each file
for g1 in range(len(gfiles)):
    # Create a CSV file for storing the embeddings
    with open(f"embs{g1}.csv", "w") as f:
        f.write("combined\n")

    # Read the content of the PDF file using pdfplumber
    content = ""
    with pdfplumber.open(gfiles[g1]) as pdf:
        for page in pdf.pages:
            content += page.extract_text() + "\n\n"

    # Split the document content into chunks
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=2000, chunk_overlap=250)
    texts = text_splitter.split_text(content)

    # Define the function to get embeddings
    def get_embedding(text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

    # Create a DataFrame and store the text chunks
    df = pd.DataFrame(columns=["combined"])
    df["combined"] = texts

    # Clean the text chunks
    df["combined"] = df["combined"].apply(lambda x: '"""' + x.replace("\n", "") + '"""')

    # Save the DataFrame to CSV
    df.to_csv(f"embs{g1}.csv", index=False)

    # Add embeddings to the DataFrame
    df["embedding"] = df["combined"].apply(lambda x: get_embedding(x))

    # Save the DataFrame with embeddings to CSV
    df.to_csv(f"embs{g1}.csv", index=False)

    # Convert the embeddings to a readable format
    embs = []
    for embedding in df["embedding"]:
        e1 = embedding.strip('[]').split(',')
        e1 = [float(e.strip()) for e in e1]
        embs.append(e1)

    df["embedding"] = embs

    # Save the final DataFrame to CSV
    df.to_csv(f"embs{g1}.csv", index=False)