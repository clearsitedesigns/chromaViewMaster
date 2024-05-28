import chromadb
import pandas as pd
import streamlit as st
import argparse


#This is just a quick test utility to validate datastore

# Set up argument parser for command line arguments
parser = argparse.ArgumentParser(description='Set the Chroma DB path')
parser.add_argument('db', help='Path to the Chroma DB directory')

# Function to render collection details
def render_collection_details(collection):
    st.subheader(f"Collection: {collection.name}")

    # Get the collection data
    data = collection.get()
    df = pd.DataFrame.from_dict(data)

    # List all available columns in the collection
    st.write("Available Columns in Collection:")
    st.write(df.columns.tolist())

    # Display the fields in the collection
    st.write("Collection Fields:")
    for field, value in data.items():
        if value is not None:
            st.write(f"{field}: {len(value)}")
        else:
            st.write(f"{field}: None")

    # Display the columns available in each part of the collection
    if 'ids' in df.columns:
        st.write("Columns in 'ids':")
        st.write(pd.DataFrame(df['ids']).columns.tolist())
        
    if 'embeddings' in df.columns:
        st.write("Columns in 'embeddings':")
        if df['embeddings'] is not None and len(df['embeddings']) > 0:
            st.write(pd.DataFrame(df['embeddings']).columns.tolist())
        else:
            st.write("None")
    
    if 'metadatas' in df.columns:
        st.write("Columns in 'metadatas':")
        if df['metadatas'] is not None and len(df['metadatas']) > 0:
            st.write(pd.DataFrame(df['metadatas']).columns.tolist())
        else:
            st.write("None")
    
    if 'documents' in df.columns:
        st.write("Columns in 'documents':")
        if df['documents'] is not None and len(df['documents']) > 0:
            st.write(pd.DataFrame(df['documents']).columns.tolist())
        else:
            st.write("None")

    # Get the unique document count based on the 'url' field in metadata
    if 'metadatas' in df.columns:
        # Extract URLs from metadata
        urls = [meta.get('url') for meta in df['metadatas'] if meta.get('url') is not None]
        unique_urls = pd.Series(urls).unique()
        num_documents = len(unique_urls)
        st.write(f"Number of Documents (unique URLs): {num_documents}")
        
        # Display the unique URLs for debugging purposes
        st.write("Unique URLs:")
        st.write(unique_urls)
        
        # Display the first few documents associated with unique URLs for verification
        st.write("Sample documents associated with unique URLs:")
        sample_docs = df[df['metadatas'].apply(lambda x: x.get('url') in unique_urls)].head(10)
        st.write(sample_docs)
    else:
        st.write("The collection does not contain a 'metadatas' column.")

    # Display the detailed data for each field
    st.write("Detailed Collection Data:")
    for field, value in data.items():
        st.write(f"{field}:")
        if isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], dict):
                df_field = pd.DataFrame(value)
                st.write(df_field)
            else:
                st.write(value)
        else:
            st.write(value)

# Main function to handle the Streamlit app
def main():
    st.title("Chroma Collection Details")

    args = parser.parse_args()
    db_path = args.db

    st.write(f"DB Path: {db_path}")

    client = chromadb.PersistentClient(path=db_path)

    collections = client.list_collections()
    collection_names = [collection.name for collection in collections]

    selected_collection = st.selectbox("Select a collection", collection_names)

    if selected_collection:
        collection = client.get_collection(name=selected_collection)
        render_collection_details(collection)

if __name__ == "__main__":
    main()
