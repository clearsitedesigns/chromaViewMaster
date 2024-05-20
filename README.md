# ChromaView Master 1.0

ChromaView Master is a Streamlit-based tool designed to help you understand, visualize, and manipulate Chroma database collections. I created this tool for situations where I frequently create ad-hoc RAG (Retrieval Augmented Generation) Chroma databases, and I needed an easier way to inspect their structure and content without relying solely on an LLM.

## Features

* **Collection Overview:** View a list of collections in your Chroma database.
* **Topic Modeling:** Identify main topics within your document collections.
* **Dimensionality Reduction:** Visualize relationships between documents.
* **Similarity Search:** Find documents similar to a query.
* **Knowledge Graph Visualization:** Explore entities and relationships.
* **Tag Cloud:** See word prominence in your collections.
* **Sunburst Chart:** Visualize hierarchical data.
* **Document Length Analysis:** Analyze document size distribution.
* **Entity Co-occurrence Matrix:** Explore entity co-occurrence frequency.
* **Sentiment Analysis:** Determine document sentiment (positive, negative, neutral).

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
