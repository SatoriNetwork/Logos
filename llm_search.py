import os
import time
import redis
import chromadb
from dotenv import load_dotenv
import psycopg2
import streamlit as st
import logging

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Load environment variables
load_dotenv()

class StreamEmbeddingManager:
    def __init__(self):
        # Initialize ChromaDB
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_ef = OpenAIEmbeddingFunction(
            api_key=self.openai_api_key,
            model_name="text-embedding-3-large"
        )
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma"
        )
        self.collection_name = "satori"
        
        # PostgreSQL connection
        self.conn = psycopg2.connect(
            database=os.getenv("POSTGRESQL_DB_NAME"),
            user=os.getenv("POSTGRESQL_USER"),
            password=os.getenv("POSTGRESQL_PASSWORD"),
            host=os.getenv("POSTGRESQL_HOST"),
            port=os.getenv("POSTGRESQL_PORT")
        )
        self.cursor = self.conn.cursor()

        # # Redis connection
        # self.redis_client = redis.Redis(
        #     host=os.getenv("REDIS_HOST", "localhost"),
        #     port=os.getenv("REDIS_PORT", 6379),
        #     db=os.getenv("REDIS_DB", 0),
        #     decode_responses=True
        # )

    def get_or_create_collection(self):
        """Retrieve or create a collection in ChromaDB."""
        return self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.openai_ef,
            metadata={"hnsw:space": "cosine"}
        )

    def store_embedding(self, stream_id, text):
        """Generate and store embedding in ChromaDB."""

        # Use ChromaDB's embedding function
        collection = self.get_or_create_collection()
        collection.add(
            documents=[text],
            ids=[f"stream_{stream_id}"],
            metadatas=[{"stream_id": stream_id, "text": text}]
        )

    def process_stream_for_embedding(self, stream_id):
        """Process a stream to generate and store its embedding."""
        self.cursor.execute("""
            SELECT s.source, s.description, s.tags, sm.entity, sm.attribute
            FROM stream AS s
            LEFT JOIN stream_meta AS sm ON s.id = sm.stream_id
            WHERE s.id = %s;
        """, (stream_id,))
        result = self.cursor.fetchone()

        if not result:
            st.error(f"Stream ID {stream_id} not found.")
            return

        source, description, tags, entity, attribute = result
        text_parts = [
            f"Source: {source}", f"Description: {description}", f"Tags: {tags}"
        ]
        if entity and attribute:
            text_parts.extend([f"Entity: {entity}", f"Attribute: {attribute}"])
        merged_text = ". ".join(text_parts)

        self.store_embedding(stream_id, merged_text)

    def process_all_streams(self):
        """Process all streams and generate embeddings."""
        self.cursor.execute("SELECT id FROM stream WHERE predicting IS NULL;")
        stream_ids = self.cursor.fetchall()
        st.write(f"Found {len(stream_ids)} streams to process.")

        for idx, (stream_id,) in enumerate(stream_ids):
            st.write(f"Processing Stream ID {stream_id} ({idx+1}/{len(stream_ids)})...")
            self.process_stream_for_embedding(stream_id)
            time.sleep(1)  # Avoid rate-limiting

    def process_new_entries(self):
        """Poll for new entries in the embeddings queue and process them."""
        while True:
            try:
                # Check for new entries in the embeddings queue
                self.cursor.execute("SELECT table_name, row_id FROM embeddings_queue WHERE processed = FALSE LIMIT 1;")
                new_entry = self.cursor.fetchone()

                if new_entry:
                    table_name, row_id = new_entry

                    # Process based on the table type
                    if table_name == "stream":
                        self.process_stream_for_embedding(row_id)

                    # Mark entry as processed
                    self.cursor.execute(
                        "UPDATE embeddings_queue SET processed = TRUE WHERE table_name = %s AND row_id = %s;",
                        (table_name, row_id)
                    )
                    self.conn.commit()

                    st.success(f"Processed new entry: {table_name} (ID: {row_id}).")
                    time.sleep(1)
                else:
                    # No new entries, wait before checking again
                    st.info("No new entries found. Waiting...")
                    time.sleep(10)
            except Exception as e:
                st.error(f"Error processing new entries: {e}")
                time.sleep(10)

    def find_closest_streams(self, user_query, top_n=5):
        """Find streams most similar to a user query using ChromaDB and fetch all details from PostgreSQL, caching results in Redis."""
        # # Generate a Redis key for the user query results
        # results_cache_key = f"query_results:{user_query}:{top_n}"

        # # Check if the results are already cached in Redis
        # cached_results = self.redis_client.get(results_cache_key)
        # if cached_results:
        #     st.info("Using cached results from Redis.")
        #     return eval(cached_results)  # Convert the stored string back to a Python object

        # Generate the embedding for the query
        collection = self.get_or_create_collection()

        # Query the ChromaDB collection
        results = collection.query(
            query_texts=user_query,
            n_results=top_n
        )

        stream_ids = []

        for metadata in results["metadatas"][0]:
            stream_ids.append(metadata['stream_id'])
        # Extract stream IDs from the results

        # Query PostgreSQL for all details from stream and stream_meta tables
        placeholders = ','.join(['%s'] * len(stream_ids))
        query = f"""
            SELECT 
                s.*, sm.*
            FROM 
                stream AS s
            LEFT JOIN 
                stream_meta AS sm
            ON 
                s.id = sm.stream_id
            WHERE 
                s.id IN ({placeholders})
        """
        self.cursor.execute(query, tuple(stream_ids))
        stream_details = self.cursor.fetchall()

        # Get column names for formatting results
        stream_columns = [desc[0] for desc in self.cursor.description]

        # Process and format results
        output = []
        for i, metadata in enumerate(results["metadatas"][0]):
            stream_id = metadata["stream_id"]
            similarity = results["distances"][0][i]

            # Match PostgreSQL data for the stream_id
            stream_data = next((dict(zip(stream_columns, row)) for row in stream_details if row[0] == stream_id), None)

            if stream_data:
                output.append({
                    "rank": i + 1,
                    "stream_id": stream_id,
                    "similarity": similarity,
                    "stream_data": stream_data
                })

        # # Cache the results in Redis as a string
        # self.redis_client.set(results_cache_key, str(output))
        # self.redis_client.expire(results_cache_key, 3600)  # Set an expiration time of 1 hour

        return output


# Streamlit App
st.title("Stream Embeddings and Query Search")
manager = StreamEmbeddingManager()

menu = st.sidebar.selectbox(
    "Select an option",
    ["Process New Entries", "Process All Streams", "Process Single Stream", "Query Search"]
)

if menu == "Process New Entries":
    st.header("Process New Entries")
    if st.button("Start Processing"):
        manager.process_new_entries()

elif menu == "Process All Streams":
    st.header("Process All Streams")
    if st.button("Start Processing"):
        manager.process_all_streams()

elif menu == "Process Single Stream":
    st.header("Process Single Stream")
    stream_id = st.text_input("Enter Stream ID")
    if st.button("Process"):
        if stream_id:
            manager.process_stream_for_embedding(stream_id)
        else:
            st.error("Please enter a valid Stream ID.")

elif menu == "Query Search":
    st.header("Query Search")
    user_query = st.text_input("Enter your query")
    top_n = st.number_input("Number of results", min_value=1, max_value=20, value=5)
    if st.button("Search"):
        if user_query:
            results = manager.find_closest_streams(user_query, top_n=top_n)
            for result in results:
                st.write(f"**Rank:** {result['rank']} - **Similarity:** {result['similarity']}")
                st.write("**Stream ID:**", result['stream_id'])
                st.write("**Stream Data:**")
                st.json(result['stream_data'])  # Display all fields in JSON format
                st.write("---")
        else:
            st.error("Please enter a query.")

