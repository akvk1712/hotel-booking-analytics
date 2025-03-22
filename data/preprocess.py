import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def create_data_directory():
    """Create the data directory if it doesn't exist."""
    if not os.path.exists("data"):
        os.makedirs("data")
    print("✅ 'data' folder is ready!")

def load_and_process_data():
    """Load and process the hotel booking dataset."""
    # Load the dataset
    df = pd.read_csv("data/processed_data.csv")
    print(f"✅ Dataset loaded! Shape: {df.shape}")
    return df

def generate_embeddings(df, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings for the text data using SentenceTransformer."""
    # Initialize the model
    model = SentenceTransformer(model_name)
    
    # Process embeddings in batches
    batch_size = 250
    num_batches = len(df) // batch_size + 1
    embeddings_list = []

    print(f"✅ Processing {num_batches} batches...")

    for i in range(num_batches):
        batch_texts = df['query_text'][i * batch_size:(i + 1) * batch_size].tolist()
        
        # Skip empty batches
        if len(batch_texts) == 0:
            print(f"⚠️ Skipping empty batch {i+1}")
            continue

        print(f"   🔄 Processing batch {i+1}/{num_batches}...")

        batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)

        if batch_embeddings.size == 0:  # Check if embeddings are empty
            print(f"⚠️ Warning: Empty embeddings for batch {i+1}, skipping...")
            continue

        embeddings_list.append(batch_embeddings)

    # Convert list of arrays to a single NumPy array
    if embeddings_list:  # Ensure there are valid embeddings before stacking
        embeddings = np.vstack(embeddings_list)
    else:
        raise ValueError("❌ Error: No embeddings generated!")

    return embeddings

def create_and_save_faiss_index(embeddings):
    """Create and save FAISS index for fast similarity search."""
    # Create and save FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save the index and embeddings
    faiss.write_index(index, "faiss_index.bin")
    np.save("embeddings.npy", embeddings)
    
    print("✅ FAISS index created and saved!")
    return index

def verify_files():
    """Verify that all necessary files have been created."""
    required_files = [
        "faiss_index.bin",
        "embeddings.npy",
        "data/processed_data.csv"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")

def main():
    """Main preprocessing pipeline."""
    try:
        # Create data directory
        create_data_directory()
        
        # Load and process data
        df = load_and_process_data()
        
        # Generate embeddings
        embeddings = generate_embeddings(df)
        
        # Create and save FAISS index
        index = create_and_save_faiss_index(embeddings)
        
        # Verify all files are created
        verify_files()
        
        print("🎉✅ Preprocessing complete! Data cleaned, embeddings generated, and FAISS index saved.")
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 