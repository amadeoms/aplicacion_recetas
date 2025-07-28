import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from neo4j import GraphDatabase
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import json

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
search_term = st.text_input("Search by name or role")