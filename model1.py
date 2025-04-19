import streamlit as st
import speech_recognition as sr1q
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)








def get_voice_input():
    # Create a recognizer object
    recognizer = sr.Recognizer()
    
    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        st.write("Speak something...")
        
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)
        
        # Capture audio input
        audio = recognizer.listen(source)
        
    try:
        # Use Google Speech Recognition to convert audio to text
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
# Function to display main page
def main_page():
    st.title("InfluenceReco")
    st.subheader("Find your Influencer today")
    
    # Option to trigger voice command
    if st.button("Voice Command"):
        st.session_state.page = "voice_command_page"
    
    # Or option
    st.write("OR")
    
    # Predefined options for the niche and location
    predefined_niches = ["Fashion", "Fitness", "Travel", "Food", "Technology", "Family", "Music", "Dance","Health","Arts","Gardening","Cooking"]
    predefined_locations = ["India", "USA","UK","Australia", "China","Malaysia", "Singapore", "Sri Lanka","Africa","Dubai","Qatar","California"]

    # Multiselect for selecting predefined niches and typing custom niche
    selected_niches = st.multiselect("Select Niche", predefined_niches)

    # Range slider for follower count
    min_followers, max_followers = st.slider("Follower Count Range", min_value=0, max_value=10000000, value=(0, 10000000))

    # Multiselect for selecting predefined locations and typing custom location
    selected_locations = st.multiselect("Select Location", predefined_locations)

    # Input text field for selecting the engagement ratio
    selected_engagement = st.text_input("Enter required engagement percentage%", placeholder="e.g.,10%")

    # Button to trigger search action
    if st.button("Go"):
        # Store user input in session state
        st.session_state.selected_niches = selected_niches
        st.session_state.min_followers = min_followers
        st.session_state.max_followers = max_followers
        st.session_state.selected_locations = selected_locations
        st.session_state.selected_engagement = selected_engagement

        # Navigate to search results page
        st.session_state.page = "search_results_page"
def preprocess_text(text):
    # Remove punctuation and special characters
    text = ''.join([char for char in str(text) if char not in punctuation])  # Convert to string to handle NaN values
    # Tokenization
    tokens = word_tokenize(text)
    # Lowercasing
    tokens = [word.lower() for word in tokens]
    # Stopwords removal
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text
# Function to display search results page
def search_results_page():
    if st.button("HOME"):
        st.session_state.page = "main_page"
    # Download NLTK resources (only need to do this once)
    nltk.download('punkt')
    nltk.download('stopwords')

    # Read the CSV file into a DataFrame
    df = pd.read_csv("insta4.csv", engine="python")

    # Selecting specific columns from DataFrame 'df'
    # Selecting specific columns from DataFrame 'df'"
    df = df[["username","category","avg_engagement","followers_count","types_of_influencer","post_count", "Niche"]]
    df.reset_index(inplace=True)

    # Add a new column 'serial_number' with serial numbers starting from 1
    df['serial_number'] = df.index + 1
    #df = df[["Rank", "Account", "Title", "Link", "Category", "Audience Country", "Authentic engagement", "Engagement avg", "Description"]]
    # Filter rows where Category value is "NULL"
    df = df[df['category'] != "Null"]

    # Define stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    df['Niche'] = df['Niche'].apply(preprocess_text)


    # Define TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # TF-IDF Vectorization for text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Niche'])

    # Convert TF-IDF matrix to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Define one-hot encoder for followers_count column
    followers_count_encoder = OneHotEncoder(sparse=False)


    # One-hot encoding for followers_count
    followers_count_encoded = followers_count_encoder.fit_transform(df[['followers_count']])

    # Convert encoded array to DataFrame
    followers_count_df = pd.DataFrame(followers_count_encoded, columns=followers_count_encoder.get_feature_names_out())

    # Reorder DataFrame columns
    df = df[["index","username", "category", "avg_engagement", "followers_count","types_of_influencer","post_count", "Niche"]]

    # Define one-hot encoder for category column
    category_encoder = OneHotEncoder(sparse=False)

    # One-hot encoding for category
    category_encoded = category_encoder.fit_transform(df[['category']])

    # Convert encoded array to DataFrame
    category_df = pd.DataFrame(category_encoded, columns=category_encoder.get_feature_names_out())

    # Concatenate TF-IDF DataFrame with one-hot encoded category and followers_count DataFrames
    features_df = pd.concat([tfidf_df, category_df, followers_count_df], axis=1)
    tfidf_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Compute cosine similarity for TF-IDF vectors
    tfidf_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Compute Jaccard similarity for categorical attributes (e.g., "Category" and "Audience Country")
    #category_similarity_matrix = jaccard_score(category_encoded, category_encoded, average=None)
    #audience_country_similarity_matrix = jaccard_score(audience_country_encoded, audience_country_encoded, average=None)

    # Define weights for combining similarity scores
    text_similarity_weight = 1.0
    category_similarity_weight = 0.3
    audience_country_similarity_weight = 0.2

    # Combine similarity scores from text-based and categorical attributes into an overall similarity score
    overall_similarity_matrix = (
        tfidf_similarity_matrix * text_similarity_weight 
    )

    # Rank influencers based on their overall similarity scores
    influencer_indices = np.arange(len(df))  # Indices of influencers
    sorted_indices = np.argsort(overall_similarity_matrix, axis=1)[:, ::-1]  # Sort indices by similarity scores (descending order)
    ranked_influencers = [influencer_indices[row] for row in sorted_indices]  # Get ranked influencers
    def recommend_top_n_influencers(influencer_id, top_n=5):
        similar_influencers = ranked_influencers[influencer_id][:top_n]
        return similar_influencers
    def embed(text):
        return model(text)
    category=list(df['Niche'])
    nan_indices = [i for i, cat in enumerate(category) if isinstance(cat, float) and np.isnan(cat)]

    # Handle NaN values (example: remove NaN values)
    category_cleaned = [cat for i, cat in enumerate(category) if i not in nan_indices]

    # Preprocess data (if needed)

    # Embed categories using TensorFlow model
    embeddings = embed(category_cleaned)
    nn= NearestNeighbors(metric='cosine', algorithm='brute')
    nn.fit(embeddings)
    def recommend(text, additional_columns=None):
        # Assuming emb and nn are defined somewhere
        emb = embed([text])
        neighbors = nn.kneighbors(emb, return_distance=False)[0]
        recommended_df = df.iloc[neighbors].copy()
        
        # If additional_columns is provided, include those columns in the recommended DataFrame
        if additional_columns:
            recommended_df = recommended_df[additional_columns + ['index','username','types_of_influencer','category','post_count']]
        
        return recommended_df
    st.title("Your Results")
    st.write("Here are the search results based on your criteria:")
    # Display user input data stored in session state
    st.write("Selected Niches:", st.session_state.selected_niches)
    st.write("Follower Count Range:", f"{st.session_state.min_followers} to {st.session_state.max_followers}")
    st.write("Selected Locations:", st.session_state.selected_locations)
    st.write("Selected Engagement:", st.session_state.selected_engagement)
    st.write("info",st.session_state.info)
    follower_count_categories = {
    "nano": (0, 10000),
    "micro": (10000, 100000),
    "macro": (100000, 1000000),
    "mega": (1000000, float('inf'))
    }

    # Get follower count
    follower_count = st.session_state.max_followers  # Assuming max_followers represents the follower count

    # Determine follower count category
    category = None
    for cat, (lower, upper) in follower_count_categories.items():
        if lower <= follower_count < upper:
            category = cat
            break

    # Prepare prompt using selected input values
    prompt = f"{', '.join(st.session_state.selected_niches)} "
    prompt += f"{category} "
    prompt += f"{', '.join(st.session_state.selected_locations)}\n"

    # Print or use the prompt for further processing
    st.write(prompt)
    recommended_articles = recommend(prompt, additional_columns=['Niche'])
    recommended_articles
    i=recommended_articles['index']

    first_value = i.iloc[0]
    first_value = int(first_value)
    recommended_influencers = recommend_top_n_influencers(first_value)
    # Initialize an empty DataFrame to store recommended influencers
    recommended_influencers_df = pd.DataFrame()

    # Iterate over recommended indices
    for recommended_index in recommended_influencers:
        # Retrieve the row corresponding to the recommended index and append to the DataFrame
        recommended_influencers_df = recommended_influencers_df.append(df.iloc[recommended_index])
    username_list = recommended_influencers_df['username'].tolist()
    avg_eng_list = recommended_influencers_df['avg_engagement'].tolist()
    post_count_list = recommended_influencers_df['post_count'].tolist()
    
    def get_instagram_profile_link(username):
        return f"https://www.instagram.com/{username}/"

    # Example usage
    link_list=[]
    for i in range(5):
        profile_link = get_instagram_profile_link(username_list[i])
        link_list.append(profile_link)
    st.write(link_list)

    # Button to navigate back to the main page
    

# Function to display voice command page

def voice_command():
    info = ""  # Initialize variable to store voice input
    optional_input = ""  # Initialize variable to store optional text input
    
    # Button to navigate back to the main page
    if st.button("HOME"):
        st.session_state.page = "main_page"
    
    # Title for the voice command page
    st.title("Voice Command1")
    
    # Button to trigger voice input
    if st.button("Voice Input"):
        info = get_voice_input()  # Get voice input
        st.write("You:", info)  # Display voice input
    
    
    # Button to display voice and optional inputs
    if st.button("Go"):
        st.write("Voice Input:", info)  # Display voice input

        

# Main function to run the app
def main():
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "main_page"

    # Display the appropriate page based on session state
    if st.session_state.page == "main_page":
        main_page()
    elif st.session_state.page == "search_results_page":
        search_results_page()
    elif st.session_state.page == "voice_command_page":
        voice_command()

if __name__ == "__main__":
    main()
