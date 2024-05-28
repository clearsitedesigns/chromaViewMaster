import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pandas as pd
import streamlit as st
import argparse
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import networkx as nx
import spacy
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from textblob import TextBlob
from pyvis.network import Network
import streamlit.components.v1 as components
import numpy as np

# Load spaCy model for entity extraction
nlp = spacy.load("en_core_web_sm")

# Set Streamlit page configuration for wide layout
st.set_page_config(layout="wide")

# Define the URL for the background image
background_image_url = "static/background.jpg"

# Inject custom CSS for full-screen layout with a custom background image and reduced checkbox font size
st.markdown(
    f"""
    <style>
        body {{
            background-image: url('{background_image_url}');
        }}
        .css-18e3th9, .css-1d391kg, .css-1lcbmhc, .css-12ttj6m, .css-15zrgzn, .css-10trblm {{
            padding: 0;
        }}
        .stApp > header {{
            background-color: transparent;
        }}
        .stApp {{
            background: url("static/background.png");
            background-size: cover;
            max-width: 100%;
            padding: 0;
            margin: 0;
            background-color: transparent;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .main {{
            padding: 0;
        }}
        .reportview-container .main .block-container {{
            padding: 0;
            margin: 0;
            width: 100%;
        }}
        .css-1e5imcs {{
            font-size: 14px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

parser = argparse.ArgumentParser(description='Set the Chroma DB path to view collections')
parser.add_argument('db')

pd.set_option('display.max_columns', 4)

def preprocess_for_sunburst(df):
    if 'category' in df.columns and 'subcategory' in df.columns and 'value' in df.columns:
        df['level_1'] = df['category']
        df['level_2'] = df['subcategory']
        df['level_3'] = df['subcategory']
        return df[['level_1', 'level_2', 'level_3', 'value']]
    else:
        st.write("The DataFrame does not contain the required columns for the Sunburst chart.")
        return pd.DataFrame()

def preprocess_for_knowledge_graph(df):
    entities = []
    relationships = []
    for doc in nlp.pipe(df['documents'].astype(str)):
        for ent in doc.ents:
            entities.append(ent.text)
        for token in doc:
            if token.dep_ == 'ROOT' and token.head.dep_ != 'ROOT':
                relationships.append((token.head.text, token.text))
    entity_freq = Counter(entities)
    return entities, relationships, entity_freq

def display_unique_document_count(df):
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

def render_collection_statistics(collection):
    st.subheader("Collection Statistics")
    st.write("View statistics on the current collection.")

    # Get the collection data
    data = collection.get()
    df = pd.DataFrame.from_dict(data)

    # Get the unique document count based on the 'url' field in metadata
    display_unique_document_count(df)

    # Calculate other statistics
    doc_lengths = df['documents'].apply(len)
    avg_doc_length = doc_lengths.mean()
    median_doc_length = doc_lengths.median()
    std_doc_length = doc_lengths.std()

    most_common_words = Counter(" ".join(df['documents']).split()).most_common(10)
    most_common_words_str = ", ".join([word for word, count in most_common_words])

    entities = [ent.text for doc in nlp.pipe(df['documents'].astype(str)) for ent in doc.ents]
    most_common_entities = Counter(entities).most_common(10)
    most_common_entities_str = ", ".join([entity for entity, count in most_common_entities])

    df['sentiment'] = df['documents'].apply(lambda x: TextBlob(x).sentiment.polarity)
    sentiment_counts = df['sentiment'].value_counts(bins=3, sort=False)
    sentiment_distribution = {
        'Negative': sentiment_counts.iloc[0],
        'Neutral': sentiment_counts.iloc[1],
        'Positive': sentiment_counts.iloc[2]
    }

    # Create a dictionary with the statistics
    statistics = {
        "Number of Documents": len(df['documents']),
        "Average Document Length": avg_doc_length,
        "Median Document Length": median_doc_length,
        "Standard Deviation of Document Length": std_doc_length,
        "Most Common Words": most_common_words_str,
        "Most Common Entities": most_common_entities_str,
        "Negative Sentiment Documents": sentiment_distribution['Negative'],
        "Neutral Sentiment Documents": sentiment_distribution['Neutral'],
        "Positive Sentiment Documents": sentiment_distribution['Positive']
    }

    # Convert all values to strings to avoid conversion errors
    statistics_str = {k: str(v) for k, v in statistics.items()}

    # Display the statistics as a table
    st.table(pd.DataFrame(statistics_str.items(), columns=['Statistic', 'Value']))

# Visualization Functions
def render_topic_modeling(df, collection_name):

   # Define a Streamlit container for the slider and pie chart
    container = st.container()
    
    with container:
        num_topics = st.slider("Number of Topics", min_value=2, max_value=10, value=5, key=f"{collection_name}_num_topics", help="Select the number of topics to be discovered in the documents.")
    
    # Vectorize the documents
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(df['documents'].astype(str))
    
    # Fit LDA model
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)
    topic_probabilities = lda.transform(doc_term_matrix)

    # Create topic probability dataframe
    topic_df = pd.DataFrame(topic_probabilities, columns=[f"Topic {i+1}" for i in range(num_topics)])
    topic_df['Document'] = df['documents']

    st.subheader("Topic Modeling Results")
    st.write("Topic modeling helps identify main themes and subjects discussed in the documents.")

    # Display topic summaries with line breaks
    feature_names = vectorizer.get_feature_names_out()
    topic_summaries = []
    for i, component in enumerate(lda.components_):
        top_words = ', '.join([feature_names[idx] for idx in component.argsort()[:-10 - 1:-1]])
        topic_summaries.append(f"Topic {i+1}: {top_words}")
    st.write("Topic Summaries:")
    st.write('\n\n'.join(topic_summaries))  # Adding double line breaks for better separation

    # Prepare data for Pie chart
    topic_distribution = topic_df.iloc[:, :-1].sum().reset_index()
    topic_distribution.columns = ['Topic', 'Probability']

    # Input for key phrase or word
    key_phrase = st.text_input("#Enter a key phrase or word to determine its probability in the documents")

    if key_phrase:
        # Vectorize the documents using TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['documents'].astype(str))
        
        # Ensure the key phrase is in the vocabulary
        if key_phrase in tfidf_vectorizer.vocabulary_:
            # Calculate the probability of the key phrase appearing in the documents
            key_phrase_index = tfidf_vectorizer.vocabulary_[key_phrase]
            probabilities = tfidf_matrix[:, key_phrase_index].toarray().flatten()
            avg_probability = np.mean(probabilities)

            st.write(f"The average probability of the key phrase '{key_phrase}' appearing in the documents is {avg_probability:.2%}")
        else:
            st.write(f"The key phrase '{key_phrase}' is not found in the documents.")

    # Update and display the pie chart whenever the topic number changes
    with container:
        fig = px.pie(
            topic_distribution, 
            names='Topic', 
            values='Probability', 
            title='Topic Distribution',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig)


def render_similarity_search(df, collection_name):
    query = st.text_input("Enter a query document", key=f"{collection_name}_query", help="Enter text to search for similar documents within the collection.")
    if query:
        results = df[df['documents'].str.contains(query, case=False, na=False)]

        st.subheader("Similarity Search Results")
        st.write("Similarity search allows you to find documents within the collection that are most similar to a given query document.")
        
        for i, result in enumerate(results['documents'].head(5)):
            st.markdown(f"**Result {i+1}:**")
            st.write(result)

def render_knowledge_graph_plotly(df, collection_name):
    entities, relationships, entity_freq = preprocess_for_knowledge_graph(df)
    top_entities = entity_freq.most_common(10)
    entity_labels = [item[0] for item in top_entities]
    entity_counts = [item[1] for item in top_entities]

    G = nx.Graph()
    G.add_nodes_from(entity_labels)
    G.add_edges_from(relationships)

    pos = nx.spring_layout(G)
    for node in G.nodes():
        G.nodes[node]['pos'] = pos[node]

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'{node}\n# of connections: {len(adjacencies[1])}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Knowledge Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    st.plotly_chart(fig)
    st.write("A knowledge graph visually represents entities and their relationships extracted from the documents.")

def render_knowledge_graph_pyvis(df, collection_name):
    st.subheader("Knowledge Graph (PyVis)")
    st.write("""
    A Knowledge Graph visualizes the relationships between entities within the documents. This graph helps identify key entities and their connections, revealing hidden patterns and insights.

    **Instructions:**
    - Enter a topic or entity of interest.
    - The graph will dynamically display nodes and edges related to the entered topic or entity.
    """)

    topic = st.text_input("Enter a topic or entity to explore its connections:")

    if topic:
        # Extract entities and relationships
        entities = []
        relationships = []
        for doc in nlp.pipe(df['documents'].astype(str)):
            if topic in doc.text:
                for ent in doc.ents:
                    entities.append(ent.text)
                for token in doc:
                    if token.dep_ == 'ROOT' and token.head.dep_ != 'ROOT':
                        relationships.append((token.head.text, token.text))
        
        entity_freq = pd.Series(entities).value_counts()
        top_entities = entity_freq.head(10).index.tolist()

        G = nx.Graph()
        G.add_nodes_from(top_entities)
        G.add_edges_from(relationships)

        net = Network(height='750px', width='100%', notebook=True)
        net.from_nx(G)
        net.show_buttons(filter_=['physics'])

        html_file = f"{collection_name}_knowledge_graph.html"
        net.save_graph(html_file)

        st.subheader("Knowledge Graph Visualization")
        st.write("The graph below shows the relationships between the entered topic or entity and other entities within the documents.")
        components.html(open(html_file, 'r', encoding='utf-8').read(), height=800)


def render_tag_cloud(df):
    text = ' '.join(df['documents'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    st.subheader("Tag Cloud")
    st.write("A tag cloud visualizes the prominence of words in the documents. Words that appear larger are more frequent in the text.")
    st.image(wordcloud.to_array(), use_column_width=True)


# Treemap chart implementation
def render_treemap_chart(df, collection_name):
    st.subheader("Treemap Chart")
    st.write("A Treemap chart visualizes hierarchical data with nested rectangles. Each rectangle represents a category, and its size is proportional to its value.")

    # Prepare data for Treemap chart
    treemap_data = []
    num_topics = st.slider("Number of Topics", min_value=2, max_value=10, value=5, key=f"{collection_name}_treemap_num_topics", help="Select the number of topics to be discovered in the documents.")

    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(df['documents'].astype(str))
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)
    topic_probabilities = lda.transform(doc_term_matrix)

    topic_df = pd.DataFrame(topic_probabilities, columns=[f"Topic {i+1}" for i in range(num_topics)])
    topic_df['Document'] = df['documents']
    topic_df['Length'] = df['documents'].str.len()  # Example of additional document detail

    for index, row in topic_df.iterrows():
        document = f"Document {index+1}"
        for topic in range(num_topics):
            probability = row[f'Topic {topic+1}']
            treemap_data.append({
                'Document': document, 
                'Topic': f'Topic {topic+1}', 
                'Probability': probability,
                'Length': row['Length'],  # Example of additional document detail
                'Text': row['Document'][:100] + '...'  # Displaying first 100 characters of the document
            })

    treemap_df = pd.DataFrame(treemap_data)

    fig = px.treemap(
        treemap_df, 
        path=['Document', 'Topic'], 
        values='Probability', 
        title='Document-Topic Probability Distribution', 
        color='Probability', 
        color_continuous_scale='Viridis',
        hover_data={'Length': True, 'Text': True}  # Adding additional details to tooltips
    )
    st.plotly_chart(fig)


def render_document_length_analysis(df):
    df['document_length'] = df['documents'].apply(len)
    st.subheader("Document Length Analysis")
    st.write("Analyze the length of the documents to understand the distribution of document sizes.")
    
    st.markdown("""
    **Detailed Explanation of Enhancements**

    **Document Length Statistics:**

    - **Mean Length**: Shows the average length of the documents. This helps understand the typical size of the documents.
    - **Median Length**: Indicates the middle value of document lengths, giving insight into the central tendency and whether the data is skewed.
    - **Mode Length**: Represents the most common document length, useful for identifying frequently occurring lengths.
    - **Min and Max Length**: Highlight the range of document lengths, indicating the shortest and longest documents.
    - **Standard Deviation**: Measures the variation in document lengths, helping to understand the consistency or diversity in document sizes.

    **Length Percentiles:**

    - **Percentiles (25th, 50th, 75th)**: Provide a clearer picture of the distribution of document lengths. For example, the 50th percentile (median) shows the midpoint, while the 25th and 75th percentiles indicate the spread of the middle half of the data.

    **Histograms and Box Plots:**

    - **Histogram**: Visualizes the distribution of document lengths, showing how frequently different lengths occur. This helps in identifying patterns, such as whether most documents are short, long, or of moderate length.
    - **Box Plot**: Displays the spread and outliers in document lengths. It highlights the median, quartiles, and potential outliers, providing a summary of the data distribution.

    **Correlation Analysis:**

    - **Correlation Matrix**: Analyzes the relationship between document length and topic probabilities. This helps in understanding whether longer documents tend to cover more topics or have higher probabilities for certain topics.
    - **Heatmap**: Visualizes the correlation matrix, making it easier to identify strong positive or negative correlations between document length and topic probabilities.
    """)

    # Basic Statistics
    mean_length = df['document_length'].mean()
    median_length = df['document_length'].median()
    mode_length = df['document_length'].mode()[0]
    min_length = df['document_length'].min()
    max_length = df['document_length'].max()
    std_length = df['document_length'].std()
    
    # Length Percentiles
    percentiles = df['document_length'].quantile([0.25, 0.5, 0.75]).to_dict()

    # Create a table of statistics
    statistics_table = pd.DataFrame({
        "Statistic": ["Mean Length", "Median Length", "Mode Length", "Min Length", "Max Length", "Standard Deviation", "25th Percentile", "50th Percentile (Median)", "75th Percentile"],
        "Value": [f"{mean_length:.2f}", f"{median_length:.2f}", f"{mode_length}", f"{min_length}", f"{max_length}", f"{std_length:.2f}", f"{percentiles[0.25]:.2f}", f"{percentiles[0.5]:.2f}", f"{percentiles[0.75]:.2f}"]
    })

    st.table(statistics_table)

    # Histogram
    fig_hist = px.histogram(df, x='document_length', nbins=50, title="Document Length Distribution")
    st.plotly_chart(fig_hist)
    
    # Box Plot
    fig_box = px.box(df, y='document_length', title="Document Length Box Plot")
    st.plotly_chart(fig_box)

    # Correlation with Topic Probabilities
    num_topics = st.slider("Number of Topics for Correlation Analysis", min_value=2, max_value=10, value=5)
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(df['documents'].astype(str))
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)
    topic_probabilities = lda.transform(doc_term_matrix)
    
    for topic in range(num_topics):
        df[f'Topic_{topic+1}_probability'] = topic_probabilities[:, topic]
    
    # Correlation Matrix
    correlation_matrix = df[['document_length'] + [f'Topic_{i+1}_probability' for i in range(num_topics)]].corr()
    st.write("Correlation Matrix")
    st.dataframe(correlation_matrix)
    
    fig_corr = px.imshow(correlation_matrix, title="Correlation Matrix Heatmap")
    st.plotly_chart(fig_corr)

def render_entity_cooccurrence_matrix(df):
    st.subheader("Entity Co-occurrence Matrix")
    st.write("""
    An entity co-occurrence matrix shows the frequency with which entities (e.g., names, places, organizations) appear together within the documents. This analysis helps to identify relationships and patterns between entities, revealing hidden connections and important groupings within the text. Please note this will take a moment to process.
    
    **Improvements and Enhancements:**

    - **Co-occurrence Matrix Visualization**: Visualize the matrix using heatmaps to easily identify strong co-occurrences.
    - **Entity Frequency Statistics**: Display the most common entities and their frequencies.
    - **Interactive Filtering**: Allow filtering by entity type or frequency thresholds.
    - **Graphical Representations**: Complement the matrix with network graphs to visualize entity relationships dynamically.
    """)

    # Extract entities and their co-occurrences
    entities = [ent.text for doc in nlp.pipe(df['documents'].astype(str)) for ent in doc.ents]
    entity_cooc = pd.crosstab(pd.Series(entities), pd.Series(entities))
    
    # Most common entities
    entity_freq = pd.Series(entities).value_counts().reset_index()
    entity_freq.columns = ['Entity', 'Frequency']

    st.write("Most Common Entities:")
    st.table(entity_freq.head(10))

    # Co-occurrence Matrix Heatmap
    fig_cooc = px.imshow(entity_cooc, title='Entity Co-occurrence Matrix', labels=dict(x="Entity", y="Entity", color="Co-occurrence Frequency"))
    st.plotly_chart(fig_cooc)

    # Network Graph for entity relationships
    G = nx.Graph()
    for i, row in entity_cooc.iterrows():
        for j, value in row.items():
            if value > 0:
                G.add_edge(i, j, weight=value)

    pos = nx.spring_layout(G, k=0.5)
    edge_x = []
    edge_y = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'{node}<br># of connections: {len(G[node])}')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2),
        text=node_text)

    node_adjacencies = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))

    node_trace.marker.color = node_adjacencies

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Knowledge Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    st.plotly_chart(fig)

def render_sentiment_analysis(df):
    df['sentiment'] = df['documents'].apply(lambda x: TextBlob(x).sentiment.polarity)
    st.subheader("Sentiment Analysis")
    st.write("""
    Perform sentiment analysis on the documents and visualize the results. Sentiment values range from -1 (negative) to 1 (positive).
    
    **Detailed Explanation of Enhancements**

    **Sentiment Score Statistics:**

    - **Mean Sentiment**: Shows the average sentiment score of the documents, providing an overall sense of positivity or negativity.
    - **Median Sentiment**: Indicates the middle value of sentiment scores, helping to understand the central tendency of sentiment.
    - **Mode Sentiment**: Represents the most frequently occurring sentiment score, useful for identifying the most common sentiment.
    - **Min and Max Sentiment**: Highlight the range of sentiment scores, indicating the most negative and most positive sentiments.
    - **Standard Deviation**: Measures the variation in sentiment scores, helping to understand the consistency or diversity in sentiment.

    **Sentiment Distribution Visualization:**

    - **Histogram**: Visualizes the distribution of sentiment scores, showing how frequently different sentiment scores occur. This helps in identifying patterns, such as whether most documents are positive, negative, or neutral.
    - **Density Plot**: Provides a smooth distribution of sentiment scores, highlighting the density of scores across the range.

    **Sentiment Category Distribution:**

    - **Pie Chart**: Shows the proportion of positive, neutral, and negative documents, giving a clear overview of sentiment categories.

    **Sentiment Word Clouds:**

    - **Word Clouds**: Generate separate word clouds for positive, neutral, and negative documents to highlight commonly used words in each category.

    **Sentiment Over Time:**

    - **Time Series**: If the documents have timestamps, visualize sentiment trends over time using line charts to understand how sentiment evolves.

    **Sentiment vs. Document Length:**

    - **Scatter Plot**: Analyze the correlation between document length and sentiment scores to understand how document size might influence sentiment.
    """)
    
    # Basic Sentiment Statistics
    mean_sentiment = df['sentiment'].mean()
    median_sentiment = df['sentiment'].median()
    mode_sentiment = df['sentiment'].mode()[0]
    min_sentiment = df['sentiment'].min()
    max_sentiment = df['sentiment'].max()
    std_sentiment = df['sentiment'].std()
    
    # Create a table of sentiment statistics
    sentiment_statistics_table = pd.DataFrame({
        "Statistic": ["Mean Sentiment", "Median Sentiment", "Mode Sentiment", "Min Sentiment", "Max Sentiment", "Standard Deviation"],
        "Value": [f"{mean_sentiment:.2f}", f"{median_sentiment:.2f}", f"{mode_sentiment:.2f}", f"{min_sentiment:.2f}", f"{max_sentiment:.2f}", f"{std_sentiment:.2f}"]
    })

    st.table(sentiment_statistics_table)

    # Sentiment Distribution Histogram
    fig_sent_hist = px.histogram(df, x='sentiment', nbins=50, title="Sentiment Distribution")
    st.plotly_chart(fig_sent_hist)
    
    # Density Plot
    fig_density = px.density_contour(df, x='sentiment', title="Sentiment Density Plot")
    st.plotly_chart(fig_density)

    # Sentiment Category Distribution
    df['sentiment_category'] = pd.cut(df['sentiment'], bins=[-1, -0.5, 0.5, 1], labels=['Negative', 'Neutral', 'Positive'])
    sentiment_counts = df['sentiment_category'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig_pie = px.pie(sentiment_counts, values='Count', names='Sentiment', title='Sentiment Category Distribution')
    st.plotly_chart(fig_pie)

    # Sentiment Word Clouds
    st.write("Word Clouds for Sentiment Categories")
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        text = ' '.join(df[df['sentiment_category'] == sentiment]['documents'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        st.image(wordcloud.to_array(), caption=f"{sentiment} Sentiment Word Cloud", use_column_width=True)

    # Sentiment Over Time (if timestamps are available)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        fig_time_series = px.line(df, x='timestamp', y='sentiment', title="Sentiment Over Time")
        st.plotly_chart(fig_time_series)
    
    # Sentiment vs. Document Length
    fig_sent_len = px.scatter(df, x='document_length', y='sentiment', title="Sentiment vs. Document Length")
    st.plotly_chart(fig_sent_len)

def render_network_centrality(G, collection_name):
    centrality = nx.degree_centrality(G)
    st.subheader("Network Centrality Metrics")
    st.write("Displays the centrality metrics for nodes in the knowledge graph. Nodes with higher centrality are more connected within the graph.")
    st.write(pd.DataFrame.from_dict(centrality, orient='index', columns=['Centrality']))

def view_collections(dir):
    st.header("ChromaView Master 1.1")
    st.markdown("DB Path: %s" % dir)
    st.write("Below are the collections found in the specified Chroma DB path. You can explore the collections by performing various actions like topic modeling, similarity search, and knowledge graph visualization. Select a chroma collection below. All tools were massively upgraded. Some may take longer to load.")

    version = "1.1"  # Replace with your actual versioning logic

    st.markdown(f"""
    **ChromaView Master {version} empowers data scientists and analysts to gain a comprehensive understanding of their Chroma DB collections.**

    Leveraging techniques like Latent Dirichlet Allocation (LDA) for topic modeling and spaCy for entity extraction, the tool provides:

    * **Advanced Analysis:** Go beyond basic statistics with in-depth visualizations.
    * **Customizable Exploration:** Choose the analysis methods that best suit your needs.
    * **Interactive Interface:** Streamlit-based for easy navigation and exploration.
    * **Open Source:** Built with flexibility and extensibility in mind. 
    """)

    client = chromadb.PersistentClient(path=dir)
    collections = client.list_collections()
    collection_names = [collection.name for collection in collections]
    selected_collection_name = st.selectbox("**Begin By Selecting a Collection**", collection_names)

    if selected_collection_name:
        for collection in collections:
            if collection.name == selected_collection_name:
                current_collection = collection  # Store the actual collection object
                data = current_collection.get()
                ids = data['ids']
                embeddings = data["embeddings"]
                metadata = data["metadatas"]
                documents = data["documents"]
                
                df = pd.DataFrame.from_dict(data)
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.subheader("Options")
                    view_collection_statistics = st.checkbox("View Collection Statistics", key=f"{collection.name}_collection_statistics", help="View statistics on the current collection.")
                    perform_topic_modeling = st.checkbox("Perform Topic Modeling", key=f"{collection.name}_topic_modeling", help="Analyze the main topics discussed in the documents.")
                    perform_similarity_search = st.checkbox("Perform Similarity Search", key=f"{collection.name}_similarity_search", help="Find documents similar to a given query.")
                    visualize_knowledge_graph = st.checkbox("Visualize Knowledge Graph (Plotly)", key=f"{collection.name}_knowledge_graph_plotly", help="Visualize the entities and their relationships extracted from the documents using Plotly.")
                    visualize_knowledge_graph_pyvis = st.checkbox("Visualize Knowledge Graph (PyVis)", key=f"{collection.name}_knowledge_graph_pyvis", help="Visualize the entities and their relationships extracted from the documents using PyVis.")
                    generate_tag_cloud = st.checkbox("Generate Tag Cloud", key=f"{collection.name}_tag_cloud", help="Visualize the prominence of words in the documents.")
                    generate_sunburst_chart = st.checkbox("Generate Treemap Probability", key=f"{collection.name}_sunburst_chart", help="Visualize the hierarchical structure of the data using a sunburst chart.")
                    analyze_document_length = st.checkbox("Analyze Document Length", key=f"{collection.name}_document_length", help="Analyze the length of the documents.")
                    display_entity_cooccurrence = st.checkbox("Display Entity Co-occurrence", key=f"{collection.name}_entity_cooccurrence", help="Display the co-occurrence matrix of entities.")
                    perform_sentiment_analysis = st.checkbox("Perform Sentiment Analysis", key=f"{collection.name}_sentiment_analysis", help="Analyze the sentiment of the documents.")
                
                with col2:
                    st.subheader("Results")
                    st.write("Your output from the left selection will appear here.")
                
                    if view_collection_statistics:
                        render_collection_statistics(collection)

                    if perform_topic_modeling:
                        render_topic_modeling(df, collection.name)
                    
                    if perform_similarity_search:
                        render_similarity_search(df, collection.name)
                    
                    if visualize_knowledge_graph:
                        render_knowledge_graph_plotly(df, collection.name)

                    if visualize_knowledge_graph_pyvis:
                        render_knowledge_graph_pyvis(df, collection.name)

                    if generate_tag_cloud:
                        render_tag_cloud(df)

                    if generate_sunburst_chart:
                        render_treemap_chart(df, collection.name)

                    if analyze_document_length:
                        render_document_length_analysis(df)

                    if display_entity_cooccurrence:
                        render_entity_cooccurrence_matrix(df)

                    if perform_sentiment_analysis:
                        render_sentiment_analysis(df)

if __name__ == "__main__":
    try:
        args = parser.parse_args()
        print("Opening database: %s" % args.db)
        view_collections(args.db)
    except:
        pass
