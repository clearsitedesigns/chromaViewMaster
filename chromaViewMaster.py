import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pandas as pd
import streamlit as st
import argparse
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
import spacy
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from textblob import TextBlob
from pyvis.network import Network
import streamlit.components.v1 as components




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

        .css-18e3th9 {{
            padding: 0;
        }}
        .css-1d391kg {{
            padding: 0;
        }}
        .css-1lcbmhc {{
            max-width: 100%;
        }}
        .css-12ttj6m {{
            max-width: 100%;
            padding: 0;
        }}
        .css-15zrgzn {{
            max-width: 100%;
            padding: 0;
        }}
        .css-10trblm {{
            max-width: 100%;
            padding: 0;
        }}
         .stApp > header {{
    background-color: transparent;
   }}
                   .stApp {{
       
             background: url("static/background.png");
             background-size: cover
        
            max-width: 100%;
            padding: 0;
            margin: 0;
            background-color:transparent;
            
            background-size: cover;
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
        }} /* Reduce the font size of checkbox labels */
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

# Updated function to render collection statistics
def render_collection_statistics(df):
    st.subheader("Collection Statistics")
    st.write("View statistics on the current collection.")
    
    num_documents = len(df)
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

    statistics = {
        "Number of Documents": num_documents,
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
    
    st.table(pd.DataFrame(statistics_str.items(), columns=['Statistic', 'Value']))

# Visualization Functions
def render_topic_modeling(df, collection_name):
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

    # Display topic summaries
    feature_names = vectorizer.get_feature_names_out()
    topic_summaries = []
    for i, component in enumerate(lda.components_):
        top_words = ', '.join([feature_names[idx] for idx in component.argsort()[:-10 - 1:-1]])
        topic_summaries.append(f"Topic {i+1}: {top_words}")
    st.write("Topic Summaries:")
    st.write('\n'.join(topic_summaries))

    # Create heatmap with improved color scheme and tooltips
    fig = px.imshow(topic_df.set_index('Document').iloc[:, :-1], labels={'color': 'Probability'}, title='Document-Topic Probability Distribution', color_continuous_scale='Viridis', aspect="auto")
    fig.update_traces(hovertemplate='Document: %{y}<br>Topic: %{x}<br>Probability: %{z:.2f}')
    fig.update_xaxes(side="top")
    fig.update_layout(xaxis_title="Topics", yaxis_title="Documents", coloraxis_colorbar=dict(title="Probability"))
    st.plotly_chart(fig)

def render_similarity_search(df, collection_name):
    query = st.text_input("Enter a query document", key=f"{collection_name}_query", help="Enter text to search for similar documents within the collection.")
    if query:
        # Reverting to the previously working code
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
    entities, relationships, entity_freq = preprocess_for_knowledge_graph(df)
    top_entities = entity_freq.most_common(10)
    entity_labels = [item[0] for item in top_entities]

    G = nx.Graph()
    G.add_nodes_from(entity_labels)
    G.add_edges_from(relationships)

    net = Network(height='750px', width='100%', notebook=True)
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])
    
    html_file = f"{collection_name}_knowledge_graph.html"
    net.save_graph(html_file)
    
    st.subheader("Knowledge Graph (PyVis)")
    st.write("A knowledge graph visually represents entities and their relationships extracted from the documents.")
    components.html(open(html_file, 'r', encoding='utf-8').read(), height=800)

def render_tag_cloud(df):
    text = ' '.join(df['documents'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    st.subheader("Tag Cloud")
    st.write("A tag cloud visualizes the prominence of words in the documents. Words that appear larger are more frequent in the text.")
    st.image(wordcloud.to_array(), use_column_width=True)

def render_sunburst_chart(df):
    st.subheader("Sunburst Chart")
    st.write("A Sunburst chart visualizes hierarchical data spanning outwards radially from root to leaves.")
    sunburst_df = preprocess_for_sunburst(df)
    if not sunburst_df.empty:
        fig = px.sunburst(sunburst_df, path=['level_1', 'level_2', 'level_3'], values='value', title="Sunburst Chart")
        st.plotly_chart(fig)
    else:
        st.write("The DataFrame does not contain the required columns for the Sunburst chart.")

def render_document_length_analysis(df):
    df['document_length'] = df['documents'].apply(len)
    st.subheader("Document Length Analysis")
    st.write("Analyze the length of the documents to understand the distribution of document sizes.")
    fig = px.histogram(df, x='document_length', nbins=50, title="Document Length Distribution")
    st.plotly_chart(fig)

def render_entity_cooccurrence_matrix(df):
    entities = [ent.text for doc in nlp.pipe(df['documents'].astype(str)) for ent in doc.ents]
    entity_cooc = pd.crosstab(pd.Series(entities), pd.Series(entities))
    st.subheader("Entity Co-occurrence Matrix")
    st.write("Shows the frequency with which entities co-occur within the documents.")
    fig = px.imshow(entity_cooc, title='Entity Co-occurrence Matrix')
    st.plotly_chart(fig)

def render_sentiment_analysis(df):
    df['sentiment'] = df['documents'].apply(lambda x: TextBlob(x).sentiment.polarity)
    st.subheader("Sentiment Analysis")
    st.write("Perform sentiment analysis on the documents and visualize the results. Sentiment values range from -1 (negative) to 1 (positive).")

    df['sentiment_category'] = pd.cut(df['sentiment'], bins=[-1, -0.5, 0.5, 1], labels=['Negative', 'Neutral', 'Positive'])

    fig = px.histogram(df, x='sentiment', nbins=50, title="Sentiment Distribution")
    fig.add_annotation(x=-0.75, y=0, text='Negative', showarrow=False, yshift=10, font=dict(size=12, color='red'))
    fig.add_annotation(x=0, y=0, text='Neutral', showarrow=False, yshift=10, font=dict(size=12, color='orange'))
    fig.add_annotation(x=0.75, y=0, text='Positive', showarrow=False, yshift=10, font=dict(size=12, color='green'))
    
    st.plotly_chart(fig)

    sentiment_counts = df['sentiment_category'].value_counts()
    st.write("Sentiment Category Counts:")
    st.write(sentiment_counts)

def render_network_centrality(G, collection_name):
    centrality = nx.degree_centrality(G)
    st.subheader("Network Centrality Metrics")
    st.write("Displays the centrality metrics for nodes in the knowledge graph. Nodes with higher centrality are more connected within the graph.")
    st.write(pd.DataFrame.from_dict(centrality, orient='index', columns=['Centrality']))

def view_collections(dir):
    st.header("ChromaView Master 1.0")
    st.markdown("DB Path: %s" % dir)
    st.write("Below are the collections found in the specified Chroma DB path. You can explore the collections by performing various actions like topic modeling, similarity search, and knowledge graph visualization. Select a chroma collection below.")
    client = chromadb.PersistentClient(path=dir)
    
    collections = client.list_collections()
    collection_names = [collection.name for collection in collections]
    selected_collection_name = st.selectbox("**Begin By Selecting a Collection**", collection_names)
    

    if selected_collection_name:
        for collection in collections:
            if collection.name == selected_collection_name:
                data = collection.get()
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
                    generate_sunburst_chart = st.checkbox("Generate Sunburst Chart", key=f"{collection.name}_sunburst_chart", help="Visualize the hierarchical structure of the data using a sunburst chart.")
                    analyze_document_length = st.checkbox("Analyze Document Length", key=f"{collection.name}_document_length", help="Analyze the length of the documents.")
                    display_entity_cooccurrence = st.checkbox("Display Entity Co-occurrence", key=f"{collection.name}_entity_cooccurrence", help="Display the co-occurrence matrix of entities.")
                    perform_sentiment_analysis = st.checkbox("Perform Sentiment Analysis", key=f"{collection.name}_sentiment_analysis", help="Analyze the sentiment of the documents.")
                
                with col2:
                    st.subheader("Results")
                    st.write("Your output from the left selection will appear here.")
                
                    if view_collection_statistics:
                        render_collection_statistics(df)

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
                        render_sunburst_chart(df)

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
