import chromadb
from chromadb.config import Settings
import json
from typing import List, Dict, Optional
import os
import time

class VectorStorage:
    def __init__(self, persist_directory: str = "chroma_db"):
        """
        Initialize the vector storage with ChromaDB
        :param persist_directory: Directory to persist the database
        """
        self.persist_directory = persist_directory
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Create or get collections for different types of data
        self.conversation_collection = self.client.get_or_create_collection(
            name="conversations",
            metadata={"description": "Store conversation history"}
        )
        
        self.reference_collection = self.client.get_or_create_collection(
            name="references",
            metadata={"description": "Store reference materials and links"}
        )

    def add_conversation(self, 
                        text: str, 
                        speaker: str, 
                        metadata: Optional[Dict] = None) -> str:
        """
        Add a conversation entry to the database
        :param text: The text content of the conversation
        :param speaker: The speaker's identifier
        :param metadata: Additional metadata about the conversation
        :return: ID of the added entry
        """
        if not metadata:
            metadata = {}
            
        # Create a unique ID based on timestamp
        entry_id = f"conv_{int(time.time() * 1000)}"
        
        # Add speaker and timestamp to metadata
        metadata.update({
            "speaker": speaker,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        self.conversation_collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[entry_id]
        )
        
        return entry_id

    def add_reference(self, 
                     content: str, 
                     url: str, 
                     metadata: Optional[Dict] = None) -> str:
        """
        Add a reference entry to the database
        :param content: The content of the reference
        :param url: The URL or source of the reference
        :param metadata: Additional metadata about the reference
        :return: ID of the added entry
        """
        if not metadata:
            metadata = {}
            
        # Create a unique ID based on timestamp
        entry_id = f"ref_{int(time.time() * 1000)}"
        
        # Add URL and timestamp to metadata
        metadata.update({
            "url": url,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        self.reference_collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[entry_id]
        )
        
        return entry_id

    def search_conversations(self, 
                           query: str, 
                           n_results: int = 5) -> List[Dict]:
        """
        Search for relevant conversations
        :param query: The search query
        :param n_results: Number of results to return
        :return: List of relevant conversations with metadata
        """
        results = self.conversation_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'id': results['ids'][0][i]
            })
            
        return formatted_results

    def search_references(self, 
                         query: str, 
                         n_results: int = 5) -> List[Dict]:
        """
        Search for relevant references
        :param query: The search query
        :param n_results: Number of results to return
        :return: List of relevant references with metadata
        """
        results = self.reference_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'id': results['ids'][0][i]
            })
            
        return formatted_results

    def get_recent_conversations(self, 
                               limit: int = 10) -> List[Dict]:
        """
        Get the most recent conversations
        :param limit: Number of conversations to retrieve
        :return: List of recent conversations with metadata
        """
        results = self.conversation_collection.get(
            limit=limit,
            where={}
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'])):
            formatted_results.append({
                'text': results['documents'][i],
                'metadata': results['metadatas'][i],
                'id': results['ids'][i]
            })
            
        return formatted_results

    def clear_all(self):
        """Clear all collections in the database"""
        self.client.delete_collection("conversations")
        self.client.delete_collection("references")
        
        # Recreate collections
        self.conversation_collection = self.client.create_collection(
            name="conversations",
            metadata={"description": "Store conversation history"}
        )
        self.reference_collection = self.client.create_collection(
            name="references",
            metadata={"description": "Store reference materials and links"}
        )
