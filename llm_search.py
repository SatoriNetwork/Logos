from fastapi import HTTPException
import os
import time
import psycopg2
import logging
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class StreamEmbeddingManager:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_ef = OpenAIEmbeddingFunction(
            api_key=self.openai_api_key,
            model_name="text-embedding-3-large"
        )
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma"
        )
        self.collection_name = "satori"
        
        self.conn = psycopg2.connect(
            database=os.getenv("POSTGRESQL_DB_NAME"),
            user=os.getenv("POSTGRESQL_USER"),
            password=os.getenv("POSTGRESQL_PASSWORD"),
            host=os.getenv("POSTGRESQL_HOST"),
            port=os.getenv("POSTGRESQL_PORT")
        )
        self.cursor = self.conn.cursor()

    def get_or_create_collection(self):
        """Retrieve or create a collection in ChromaDB."""
        return self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.openai_ef,
            metadata={"hnsw:space": "cosine"}
        )

    def store_embedding(self, stream_id, text):
        """Generate and store embedding in ChromaDB."""
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
            raise HTTPException(status_code=404, detail=f"Stream ID {stream_id} not found.")

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
        for (stream_id,) in stream_ids:
            self.process_stream_for_embedding(stream_id)
            time.sleep(1) 

    def find_closest_streams(self, user_query, top_n=5):
        """Find streams most similar to a user query using ChromaDB."""
        collection = self.get_or_create_collection()
        results = collection.query(
            query_texts=user_query,
            n_results=top_n
        )
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
    
    def format_data_for_api(self, user_query: str, stream: dict) -> str:
        """
        Format the user query and stream data into a structured prompt for OpenAI's API.
        
        Returns:
            str: A formatted prompt.
        """
        prompt = (
            f"The user asked: '{user_query}'.\n\n"
            "Here is the data stream:\n"
            f"{stream}\n\n"
            "Explain the relationship between the user's query and the data stream, "
            "focusing on trends, averages, and relevance to the query. Provide actionable insights."
        )
        return prompt
    
    def fetch_explanation(self, user_query: str, stream: dict) -> dict:
        """
        Use OpenAI's API to generate an explanation of the relationship.
        
        Returns:
            dict: The API response containing the explanation.
        """
        try:
            prompt = self.format_data_for_api(user_query, stream)
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are assistant to explain the relationship between user's query and data stream"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            
            return {
                "response": response.choices[0].message.content.strip()
            }
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
        for stream in streams:
            explanation = self.fetch_explanation(user_query, stream)
            if explanation:
                explanation_with_stream_id = {
                    "stream_id": stream["stream_id"],
                    "explanation": explanation
                }
                explanations.append(explanation_with_stream_id)
        
        return explanations