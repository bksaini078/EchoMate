from typing import List, Dict, Optional
from .vectordb import VectorStorage
import time

class ContextManager:
    def __init__(self, vector_storage: VectorStorage):
        """
        Initialize the context manager
        :param vector_storage: Instance of VectorStorage for persistence
        """
        self.vector_storage = vector_storage
        self.current_context = {
            'participants': set(),
            'current_topic': None,
            'recent_messages': [],
            'active_references': []
        }
        self.max_recent_messages = 10

    def add_message(self, 
                   text: str, 
                   speaker: str, 
                   metadata: Optional[Dict] = None) -> str:
        """
        Add a new message to the context and storage
        :param text: Message content
        :param speaker: Speaker identifier
        :param metadata: Additional metadata
        :return: Message ID
        """
        # Add to vector storage
        message_id = self.vector_storage.add_conversation(text, speaker, metadata)
        
        # Update current context
        self.current_context['participants'].add(speaker)
        self.current_context['recent_messages'].append({
            'id': message_id,
            'text': text,
            'speaker': speaker,
            'timestamp': time.time()
        })
        
        # Maintain recent messages limit
        if len(self.current_context['recent_messages']) > self.max_recent_messages:
            self.current_context['recent_messages'].pop(0)
            
        return message_id

    def add_reference(self, 
                     content: str, 
                     url: str, 
                     metadata: Optional[Dict] = None) -> str:
        """
        Add a reference to the context and storage
        :param content: Reference content
        :param url: Reference URL
        :param metadata: Additional metadata
        :return: Reference ID
        """
        # Add to vector storage
        ref_id = self.vector_storage.add_reference(content, url, metadata)
        
        # Add to active references
        self.current_context['active_references'].append({
            'id': ref_id,
            'url': url,
            'timestamp': time.time()
        })
        
        return ref_id

    def get_relevant_context(self, query: str, max_results: int = 5) -> Dict:
        """
        Get relevant context for a given query
        :param query: Search query
        :param max_results: Maximum number of results to return
        :return: Dictionary containing relevant context
        """
        # Search conversations and references
        relevant_conversations = self.vector_storage.search_conversations(
            query, 
            n_results=max_results
        )
        relevant_references = self.vector_storage.search_references(
            query, 
            n_results=max_results
        )
        
        return {
            'recent_context': self.current_context['recent_messages'],
            'relevant_conversations': relevant_conversations,
            'relevant_references': relevant_references,
            'participants': list(self.current_context['participants']),
            'current_topic': self.current_context['current_topic']
        }

    def update_topic(self, topic: str):
        """
        Update the current conversation topic
        :param topic: New topic
        """
        self.current_context['current_topic'] = topic

    def clear_context(self):
        """Clear the current context"""
        self.current_context = {
            'participants': set(),
            'current_topic': None,
            'recent_messages': [],
            'active_references': []
        }

    def get_conversation_summary(self) -> Dict:
        """
        Get a summary of the current conversation
        :return: Dictionary containing conversation summary
        """
        return {
            'participants': list(self.current_context['participants']),
            'topic': self.current_context['current_topic'],
            'message_count': len(self.current_context['recent_messages']),
            'reference_count': len(self.current_context['active_references']),
            'duration': time.time() - (self.current_context['recent_messages'][0]['timestamp'] 
                                     if self.current_context['recent_messages'] 
                                     else time.time())
        }

    def get_active_references(self) -> List[Dict]:
        """
        Get currently active references
        :return: List of active references
        """
        return self.current_context['active_references']

    def remove_reference(self, ref_id: str):
        """
        Remove a reference from active references
        :param ref_id: Reference ID to remove
        """
        self.current_context['active_references'] = [
            ref for ref in self.current_context['active_references']
            if ref['id'] != ref_id
        ]
