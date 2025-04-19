
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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import streamlit as st
import os
import shutil
import tempfile
import tensorflow_hub as hub

# Define the module URL
module_url = "https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2"#"https://tfhub.dev/google/universal-sentence-encoder/4"  # or "https://tfhub.dev/google/universal-sentence-encoder-large/5"

# Load the model from TensorFlow Hub
try:
    model = hub.load(module_url)
    #print("Model loaded successfully")
except ValueError as e:
    # Create a temporary directory for the cache
    custom_cache_dir = tempfile.mkdtemp()

    # Set the environment variable for TensorFlow Hub cache
    os.environ["TFHUB_CACHE_DIR"] = custom_cache_dir

    # Clear the cache directory
    shutil.rmtree(custom_cache_dir, ignore_errors=True)
    print("Cleared TensorFlow Hub cache")

    print(f"Error loading model: {e}")






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


# Define TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# TF-IDF Vectorization for text data
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Niche'])

# Convert TF-IDF matrix to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())



# Reorder DataFrame columns
df = df[["index","username", "category", "avg_engagement", "followers_count","types_of_influencer","post_count", "Niche"]]




# Optionally, concatenate with other numerical features like "Authentic engagement" and "Engagement avg"
# features_df = pd.concat([features_df, df[['Authentic engagement', 'Engagement avg']]], axis=1)
# Print or use the features DataFrame
#features_df.head(1)



# Compute cosine similarity for TF-IDF vectors
tfidf_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Compute Jaccard similarity for categorical attributes (e.g., "Category" and "Audience Country")
#category_similarity_matrix = jaccard_score(category_encoded, category_encoded, average=None)
#followers_count_similarity_matrix = jaccard_score(followers_count_encoded, followers_count_encoded, average=None)




# Compute cosine similarity for TF-IDF vectors
tfidf_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Compute Jaccard similarity for categorical attributes (e.g., "Category" and "Audience Country")
#category_similarity_matrix = jaccard_score(category_encoded, category_encoded, average=None)
#audience_country_similarity_matrix = jaccard_score(audience_country_encoded, audience_country_encoded, average=None)

# Define weights for combining similarity scores
text_similarity_weight = 1.0


# Combine similarity scores from text-based and categorical attributes into an overall similarity score
overall_similarity_matrix = (
    tfidf_similarity_matrix * text_similarity_weight 
)

# Rank influencers based on their overall similarity scores
influencer_indices = np.arange(len(df))  # Indices of influencers
sorted_indices = np.argsort(overall_similarity_matrix, axis=1)[:, ::-1]  # Sort indices by similarity scores (descending order)
ranked_influencers = [influencer_indices[row] for row in sorted_indices]  # Get ranked influencers

# Recommend the top-N influencers with the highest similarity scores
def recommend_top_n_influencers(influencer_id, top_n=20):
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
#print(embeddings.shape)
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

st.set_page_config(page_title="InfluenceReco",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.image("img2.jpg",  width=300, use_column_width=True)
st.write("<style>img {height: 100px !important;}</style>", unsafe_allow_html=True)
st.header("Influencer Recommender 🤖")
#st.subheader("A Tool for Recommending Great Influencers On Social Media for Digital Marketing")
st.write("<span style='font-size:small; color:red'>A Tool for Recommending Great Influencers On Social Media for Digital Marketing</span>", unsafe_allow_html=True)

# Predefined options for the niche and location
unique_categories = df['category'].unique()
predefined_locations = ["India", "USA","UK","Australia", "China","Malaysia", "Singapore", "Sri Lanka","Africa","Dubai","Qatar","California"]

    # Multiselect for selecting predefined niches and typing custom niche
selected_niches = st.selectbox("Select Niche", unique_categories)

    # Range slider for follower count
min_followers, max_followers = st.slider("Follower Count Range", min_value=0, max_value=10000000, value=(0, 10000000))

    # Multiselect for selecting predefined locations and typing custom location
selected_locations = st.selectbox("Select Location", predefined_locations)

    # Input text field for selecting the engagement ratio
selected_engagement = st.text_input("Enter required engagement percentage%", placeholder="e.g.,10%")
type_of_inf = st.selectbox('Type of Influencer', ('Mega 1M+', 'Macro 100k-1M', 'Micro 10k-100k', 'Nano <10k'), index=0)


    # Prepare prompt using selected input values
prompt=""
prompt = f"{selected_niches} from {selected_locations} and {type_of_inf} and engagement rate should be {selected_engagement}"

#prompt = f"{', '.join(selected_niches)} "
#prompt += f"{type_of_inf} "
#prompt += f"{', '.join(selected_locations)}\n"
print(prompt)
if st.button("GET RESULT"):
# Example usage:
    recommended_articles = recommend(prompt, additional_columns=['Niche'])
    i=recommended_articles['index']

    first_value = i.iloc[0]
    first_value = int(first_value)

    #recommended_articles

    recommended_influencers = recommend_top_n_influencers(first_value)


    recommended_influencers_df = df.iloc[recommended_influencers]

    # Extract all entries from the 'username' column and convert to a list
    username_list = recommended_influencers_df['username'].tolist()
    avg_engagement_list = recommended_influencers_df['avg_engagement'].tolist()
    post_count_list = recommended_influencers_df['post_count'].tolist()
    





    def get_instagram_profile_link(username):
        return f"https://www.instagram.com/{username}/"

  
   

    # Assuming recommended_influencers_df is your DataFrame containing influencer data
    # Assuming link_list is a list containing Instagram profile links

    # Display each influencer with their information


# Assuming recommended_influencers_df is your DataFrame containing influencer data
# Assuming link_list is a list containing Instagram profile links

# Display multiple images along with other information in a single box
    for index, row in recommended_influencers_df.iterrows():
        col1, col2 = st.columns([1, 4])
        
        with col1:
            # Display image
            st.image('user.png', caption=f"{row['username']}", width=30, use_column_width=True)
            
        with col2:
            
            # Display post count and average engagement
            st.write(f"Post Count: {row['post_count']}")
            st.write(f"Average Engagement: {row['avg_engagement']}")
            
            # Display Instagram profile link
            instagram_profile_link = f"https://www.instagram.com/{row['username']}/"
            st.write(f"Instagram Profile: [{row['username']}]({instagram_profile_link})")
    
    

     


