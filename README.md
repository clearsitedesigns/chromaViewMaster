# ChromaView Master 1.0

ChromaView Master is a Streamlit-based tool designed to help you understand, visualize, and manipulate Chroma database collections. I created this tool for situations where I frequently create ad-hoc RAG (Retrieval Augmented Generation) Chroma databases, and I needed an easier way to inspect their structure and content without relying solely on an LLM. I just needed a simple tool to quickly analyze chroma databases. 


![screen_view](https://github.com/clearsitedesigns/chromaViewMaster/assets/5733537/9d8dafcd-55da-4505-a01b-91aa972c0f29)





## Features


* **Collection Overview:** Get a quick overview of all the collections stored within your Chroma database, along with basic statistics like the number of documents in each collection.
* **Topic Modeling:** Employ Latent Dirichlet Allocation (LDA) to uncover the main themes and topics discussed within your document collections. Understand the distribution of topics across documents and identify the most representative words for each topic.
* **Dimensionality Reduction:** Utilize t-SNE to reduce the dimensionality of your document embeddings, allowing you to visualize document relationships in a 2D scatter plot. Documents closer together are semantically more similar.
* **Similarity Search:** Enter a query and find the most similar documents within your collection based on semantic similarity. This can be helpful for retrieving relevant information or identifying duplicates.
* **Knowledge Graph Visualization:** Extract entities (people, places, organizations, etc.) and their relationships from your documents and visualize them as an interactive network graph. Gain insights into the connections between different concepts in your data.
* **Tag Cloud:** Visualize the most frequent words in your collection as a word cloud, where the size of each word corresponds to its frequency. Quickly grasp the dominant themes and vocabulary used in your documents.
* **Sunburst Chart:** If your data has a hierarchical structure (e.g., categories and subcategories), visualize it using a sunburst chart. This interactive chart allows you to drill down into different levels of the hierarchy.
* **Document Length Analysis:** Analyze the distribution of document lengths within your collection. Identify outliers and understand the general characteristics of your text data.
* **Entity Co-occurrence Matrix:** Explore how often different entities co-occur within the same documents. This can reveal patterns and associations between entities.
* **Sentiment Analysis:** Perform sentiment analysis on your documents to determine whether the overall tone is positive, negative, or neutral. Understand the emotional content of your text data.

![tag_cloud_sample](https://github.com/clearsitedesigns/chromaViewMaster/assets/5733537/44a3a7e1-87ab-46fb-9485-f68eaa35c3af)

## Prerequisites

- Python 3.7 or above
- Operating System: Windows, macOS, or Linux
- Conda (recommended for creating a clean environment)


## Installation and Usage

1. **Prerequisites:**
   * Ensure you have Python and `pip` installed. (I recommend using a clean conda environment.)
   * Install required libraries:
      ```bash
      pip install chromadb streamlit pandas sklearn spacy networkx plotly wordcloud textblob plotly.express plotly.graph_objects
      ```
   * Download the spaCy model:
      ```bash
      python -m spacy download en_core_web_sm
      ```

2. **Run ChromaView Master:**
   * Locate the path to your Chroma database.
   * Execute the following command in your terminal:
      ```bash
      streamlit run chromaMaster.py /path_to_your_database/testdb 
      ```
      (Replace `/path_to_your_database/testdb` with your actual path)

**Note:** This tool currently assumes your database has no password protection.

## Example

If your Chroma database is located at `/Users/yourname/documents/mychromadb`, you would run:

```bash
streamlit run chromaMaster.py /Users/yourname/documents/mychromadb


## Troubleshooting

- Large datastores may take awhile to generate. This is a limitation of browser & streamlit.

You can find me on discord Preston McCauley
