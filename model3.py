import os
import shutil
import tensorflow_hub as hub
import tensorflow as tf
import streamlit as st

# Define the custom cache directory
custom_cache_dir = os.path.join(os.path.expanduser("~"), ".tfhub_modules")

# Set the environment variable for TensorFlow Hub cache
os.environ["TFHUB_CACHE_DIR"] = custom_cache_dir

# Clear the TensorFlow Hub cache (optional)
def clear_tfhub_cache():
    shutil.rmtree(custom_cache_dir, ignore_errors=True)
    print("Cleared TensorFlow Hub cache")

clear_tfhub_cache()

# Define the module URL
module_url ="https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2"#"https://tfhub.dev/google/universal-sentence-encoder/4"  # or "https://tfhub.dev/google/universal-sentence-encoder-large/5"

# Try loading the model from TensorFlow Hub
try:
    model = hub.load(module_url)
    print("Model loaded successfully")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Function to embed text using the model
def embed(text):
    if model is not None:
        return model(text)
    else:
        raise ValueError("Model is not loaded properly")

# Example input data for embedding
category_cleaned = ["This is an example sentence.", "Another example sentence."]

# Embed the input data
try:
    if model is not None:
        embeddings = embed(category_cleaned)
        print("Embeddings created successfully")
        print(embeddings)
    else:
        print("Model is not available for creating embeddings")
except Exception as e:
    print(f"Error creating embeddings: {e}")

# Streamlit UI
st.title("Text Embedding Example")
input_text = st.text_area("Enter text to embed:", "Type here...")

if st.button("Embed Text"):
    try:
        if model is not None:
            result = embed([input_text])
            st.write("Embedding:")
            st.write(result.numpy())
        else:
            st.error("Model is not available for embedding")
    except Exception as e:
        st.error(f"Error embedding text: {e}")
