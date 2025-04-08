import torch
import os
from clip import load

# Prevent Streamlit from inspecting torch.classes
if hasattr(torch, 'classes'):
    del torch.classes

import streamlit as st
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from pymongo import MongoClient
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv('mongo_info.env')

# MongoDB connection
# uri = os.environ.get("MONGO_URI")
uri = st.secrets["MONGO_URI"]
if uri is None:
    raise ValueError("MONGO_URI environment variable not found")
    
client = MongoClient(uri)
db = client["recipe"]
collection = db["recipe_embed"]

# Load CLIP model only once at startup using @st.cache_resource
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Try to load from cache first
    cache_dir = os.path.expanduser('~/.cache/open_clip')
    model_path = os.path.join(cache_dir, 'model_state.pt')
    
    # Load the model
    model, preprocess = load("ViT-B/16", device=device)
    
    # If cached model exists, load its state
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Loaded model from cache")
        except Exception as e:
            print(f"Error loading cached model: {e}")
            print("Using freshly loaded model instead")
    
    return model, preprocess, device

# Function to process image and get recommendations
def process_image_and_get_recommendations(image, model, preprocess, device, k_recipes):
    # Preprocess and get embedding
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        
    # Normalize the features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    # Convert to numpy array for MongoDB query
    query_embedding = image_features[0].cpu().numpy()
    
    # Vector similarity search in MongoDB using vectorSearch
    similar_recipes = collection.aggregate([
        {
            "$vectorSearch": {
                "index": "recipe_vector",
                "path": "embedding",
                "queryVector": query_embedding.tolist(),
                "numCandidates": 100,
                "limit": k_recipes
            }
        },
        {
            "$project": {
                "title": 1,
                "joined_recipe": 1,
                "score": { "$meta": "vectorSearchScore" }
            }
        }
    ])
    
    return similar_recipes

# Load model and preprocess function
model, preprocess, device = load_model()

st.title("MunchMatch")

# Add slider for number of recommendations
k_recipes = st.slider("Number of recipes to recommend", min_value=1, max_value=10, value=5)

# File uploader
uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Food Image', use_column_width=True)
    
    # Process image and get recommendations
    similar_recipes = process_image_and_get_recommendations(image, model, preprocess, device, k_recipes)

    # Display similar recipes
    st.write("Most Similar Recipes:")
    for recipe in similar_recipes:
        st.write("---")
        st.write(f"Recipe Name: {recipe.get('title', 'N/A')}")
        # Display similarity score in small font, handling the case where score might be 'N/A'
        score = recipe.get('score', 'N/A')
        if isinstance(score, (int, float)):
            score_text = f"{score:.4f}"
        else:
            score_text = str(score)
        st.markdown(f"<p style='font-size: 12px;'>Similarity Score: {score_text}</p>", unsafe_allow_html=True)
        
        # Parse the joined_recipe field
        joined_recipe = recipe.get('joined_recipe', '')
        if joined_recipe:
            # Split the joined recipe into sections
            sections = joined_recipe.split('\n\n')
            
            # Extract ingredients and instructions
            ingredients = "N/A"
            instructions = "N/A"
            
            for section in sections:
                if section.strip().startswith("Ingredients"):
                    ingredients = section.replace("Ingredients", "").strip()
                elif section.strip().startswith("Instructions"):
                    instructions = section.replace("Instructions", "").strip()
        
        st.write(f"Ingredients: {ingredients}")
        st.write(f"Instructions: {instructions}")
