from fastapi import HTTPException
import os
import time
import psycopg
import logging
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from .chromadb import CustomRemoteClient

load_dotenv()

class StreamEmbeddingManager:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_ef = OpenAIEmbeddingFunction(
            api_key=self.openai_api_key,
            model_name="text-embedding-3-large")
        self.chroma_client = CustomRemoteClient(
            host="https://159.65.144.150:443",
            api_key=os.getenv("CHROMA_API_KEY"))
        # alternative to using the remote solution we could include the chroma
        # database in the actual docker container and access it like this:
        #self.chroma_client = chromadb.PersistentClient(path="./chroma")

        self.collection_name = "satori"

        self.conn = psycopg.connect(
            dbname=os.getenv("POSTGRESQL_DB_NAME"),
            user=os.getenv("POSTGRESQL_USER"),
            password=os.getenv("POSTGRESQL_PASSWORD"),
            host=os.getenv("POSTGRESQL_HOST"),
            port=os.getenv("POSTGRESQL_PORT"))
        self.cursor = self.conn.cursor()

    def get_or_create_collection(self):
        """Retrieve or create a collection in ChromaDB."""
        return self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.openai_ef,
            metadata={"hnsw:space": "cosine"})

    def store_embedding(self, stream_id, text):
        """Generate and store embedding in ChromaDB if not already exists."""
        collection = self.get_or_create_collection()

        # Check if embedding for stream_id already exists
        existing_embeddings = collection.get(ids=[f"stream_{stream_id}"])
        if existing_embeddings['ids']:  # If IDs are returned, embedding exists
            logging.info(f"Embedding for Stream ID {stream_id} already exists. Skipping.")
            return

        # Add new embedding to the collection
        collection.add(
            documents=[text],
            ids=[f"stream_{stream_id}"],
            metadatas=[{"stream_id": stream_id, "text": text}])
        logging.info(f"Embedding for Stream ID {stream_id} added successfully.")


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
            raise HTTPException(status_code=404, detail=f"Stream ID {stream_id} not found.")
        source, description, tags, entity, attribute = result
        text_parts = [
            f"Source: {source}",
            f"Description: {description}",
            f"Tags: {tags}"]
        if entity and attribute:
            text_parts.extend([f"Entity: {entity}", f"Attribute: {attribute}"])
        merged_text = ". ".join(text_parts)
        self.store_embedding(stream_id, merged_text)

    def process_all_streams(self):
        """Process all streams and generate embeddings."""
        self.cursor.execute("SELECT id FROM stream WHERE predicting IS NULL;")
        stream_ids = self.cursor.fetchall()
        for (stream_id,) in stream_ids:
            self.process_stream_for_embedding(stream_id)
            time.sleep(1)

    def find_closest_streams(self, user_query, top_n=5):
        """Find streams most similar to a user query using ChromaDB."""
        collection = self.get_or_create_collection()
        results = collection.query(
            query_texts=user_query,
            n_results=top_n)
        stream_ids = [metadata["stream_id"] for metadata in results["metadatas"][0]]
        placeholders = ','.join(['%s'] * len(stream_ids))
        query = f"""
            SELECT
                s.*, sm.entity, sm.attribute
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
        output = []
        stream_columns = [desc[0] for desc in self.cursor.description]
        for i, metadata in enumerate(results["metadatas"][0]):
            stream_id = metadata["stream_id"]
            similarity = results["distances"][0][i]
            stream_data = next((dict(zip(stream_columns, row)) for row in stream_details if row[0] == stream_id), None)
            if stream_data:
                output.append({
                    "rank": i + 1,
                    "stream_id": stream_id,
                    "similarity": similarity,
                    "stream_data": stream_data
                })
        return output


class StreamRAGManager:
    """
    A class to manage and explain the relationship between a user query and stream data using OpenAI's API.
    """

    def __init__(self):
        """
        Initialize the OpenAI client and model name.
        """
        self.model_name = "gpt-4o"
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def format_data_for_api(self, user_query: str, streams: list) -> str:
        """
        Format the user query and stream data into a structured prompt for OpenAI's API.

        Returns:
            str: A formatted prompt.
        """
        prompt = (
            "You are the voice of a future-predicting worldwide network called Satori that analyzes data streams and generates predictions to provide users with clear, confident, and concise insights about future trends..\n\n"
            f"A user has asked you this question: '{user_query}'.\n\n"
            "Here are the data streams payload:\n"
            f"{streams}\n\n"
            "You are to provide a simple, confident, concise, and honest answer (usually some form of prediction, or insight into the future) to the user's question. "
            "The data streams payload includes descriptive details and summaries of predictions (made by the Satori network) for data stream(s) that might be relevant to the user's question. "
            "Analyze the provided data streams payload, focusing on how it can inform an answer to the user's question. "
            "The user's query may be a detailed question about the data stream(s) in which case the answer can be precise as possible, explaining trends and outcomes, and any relevant information in straightforward language. "
            "If the user’s query is unrelated to the data, you provide a logical and intuitive response while encouraging more specific questions for deeper insights, you may suggest topics (the topics that the provided the data streams payload describes) for them to ask about. "
            "You combine the data provided with your intelligent analysis and intuition to make predictions or clarify what trends suggest about the future. "
            "In short, answer the user's question as best you can. Your tone is clear, direct, concise, simple, pleasant, helpful, and honest, revealing the network's insight with user-friendly communication."
        )
        return prompt

    def fetch_explanation(self, user_query: str, streams: list) -> dict:
        """
        Use OpenAI's API to generate an explanation of the relationship.

        Returns:
            dict: The API response containing the explanation.
        """
        try:
            prompt = self.format_data_for_api(user_query, streams)
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are assistant to explain the relationship between user's query and data stream"},
                    {"role": "user", "content": prompt}],
                temperature=0.7)
            response_message = response.choices[0].message.content
            cleaned_response = ' '.join(response_message.split())
            return {"response": cleaned_response}
        except Exception as e:
            return {"error": str(e)}

    def explain_relationship(self, user_query: str, streams: list) -> list:
        """
        Generate explanations for the relationship between the user query and each data stream.

        Args:
            user_query (str): The user's query.
            streams (list): A list of data streams to analyze.

        Returns:
            list: A list of explanations for each stream.
        """
        explanations = []
        explanation = self.fetch_explanation(user_query, streams)
        if explanation:
            explanation_with_stream_id = {"explanation": explanation}
            explanations.append(explanation_with_stream_id)

        return explanations
