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

## Installation

1. **Prerequisites:** Ensure you have Python and `pip` installed. (I recommend using a clean conda environment.)
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
