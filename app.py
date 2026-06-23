import streamlit as st
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import os
import math
import numpy as np

# here pypdf, openai, dotenv are toolboxes and PdfReader, OpenAi, load_dotenv are the actual tools we are gonna use

load_dotenv()

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

st.write("Chat with pdf")

uploaded_file = st.file_uploader(
        "Upload a pdf",
        type=["pdf"]
)
all_text = ""

def create_chunks(text,chunk_size):
    i = 0
    chunks = []
    while(i<len(text)):
        chunks.append(text[i:i+chunk_size])
        i+=chunk_size
    return chunks  
  
def get_embedding(text):
    response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text
    )
    return response.data[0].embedding  

def similarity(vec1,vec2):
    score = 0

    for i in range(len(vec1)):
        score += abs(vec1[i]-vec2[i])
    return score

# def cosine_similarity(vec1,vec2):
#     dot_product = 0
#     norm1 = 0
#     norm2 = 0

#     for i in range(len(vec1)):
#         dot_product += vec1[i] * vec2[i]
#         norm1 += vec1[i] * vec1[i]
#         norm2 += vec2[i] * vec2[i]

#     return dot_product /(
#         math.sqrt(norm1) + math.sqrt(norm2)
#     )    

def cosine_similarity(vec1,vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    return np.dot(v1,v2)/(
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )

if uploaded_file:
    reader = PdfReader(uploaded_file)
   
    for page in reader.pages:
        text = page.extract_text()
        all_text += text

    # print(len(all_text))    

    chunks = create_chunks(all_text,500)    
    if "data" not in st.session_state or st.session_state.pdf_text!=all_text:
        data = []
        # st.write(len(chunks))
        # st.write("Creating embeddings...")      
        for chunk in chunks:
            embedding = get_embedding(chunk)
            data.append({"text":chunk, "embedding":embedding})
         
        st.session_state.data = data 
        st.session_state.pdf_text = all_text
        # st.write("Embeddings created!")  
        # st.write(embedding)    
    question = st.text_input(
        "Ask a question about the pdf"
    )    
    
    question_embedding = get_embedding(question)
    # best_chunk = ""
    # best_score = float('inf')

    # data = st.session_state.data
    # for chunk in data:
    #     score = similarity(chunk["embedding"],question_embedding)
    #     if score<best_score:
    #         best_score = score
    #         best_chunk = chunk["text"]

    # best_chunk = ""
    # storage = []
    # data = st.session_state.data
    # for chunk in data:
    #     score = similarity(chunk["embedding"],question_embedding)
    #     storage.append((score,chunk["text"]))

    # storage.sort()

    best_chunk = ""
    storage = []
    data = st.session_state.data

    for chunk in data:
        score = cosine_similarity(chunk["embedding"],question_embedding)
        storage.append((score,chunk["text"]))

    storage.sort(reverse=True)
    top_3 = storage[:3]
    for x in top_3:
        best_chunk += x[1]

    # for x in storage:
    #     st.write(f"score: {x[0]} , text: {x[1]}")    

    if question.strip()!="" and all_text.strip()!="":
        prompt =f"""
        you are an assistant whose purpose is to help user understand the given document 


        Relevant context:
        {best_chunk}

        Question:
        {question}

        Answer the given question using only the relevant context available here
        Be as concise and precise as u can be
        Also if u cant answer based on it just tell him so
        """
        response = client.responses.create(
            model="gpt-5.4",
            input=prompt
        )   
        st.write(response.output_text)