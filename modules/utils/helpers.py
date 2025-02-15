from typing import Dict, List, Optional, Any
import json
import time
from datetime import datetime
import re
import os

def format_timestamp(timestamp: float) -> str:
    """
    Format a timestamp into a human-readable string
    :param timestamp: Unix timestamp
    :return: Formatted string
    """
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def sanitize_text(text: str) -> str:
    """
    Clean and sanitize text input
    :param text: Input text
    :return: Sanitized text
    """
    # Remove special characters and excessive whitespace
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    text = ' '.join(text.split())
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Split text into smaller chunks for processing
    :param text: Input text
    :param chunk_size: Maximum size of each chunk
    :return: List of text chunks
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # Add 1 for space
        if current_size + word_size > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def format_references(references: List[Dict]) -> str:
    """
    Format reference links into HTML
    :param references: List of reference dictionaries
    :return: Formatted HTML string
    """
    if not references:
        return ""
        
    html = "<ul style='list-style-type: none; padding: 0;'>"
    for ref in references:
        html += f"""
        <li style='margin-bottom: 10px;'>
            <a href='{ref.get("url", "")}' target='_blank' style='text-decoration: none;'>
                <div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>
                    <div style='font-weight: bold;'>{ref.get("title", "No Title")}</div>
                    <div style='color: #666; font-size: 0.9em;'>{ref.get("snippet", "")}</div>
                </div>
            </a>
        </li>
        """
    html += "</ul>"
    return html

def format_conversation(messages: List[Dict]) -> str:
    """
    Format conversation messages into HTML
    :param messages: List of message dictionaries
    :return: Formatted HTML string
    """
    if not messages:
        return ""
        
    html = "<div style='padding: 10px;'>"
    for msg in messages:
        speaker = msg.get("speaker", "Unknown")
        text = msg.get("text", "")
        timestamp = format_timestamp(msg.get("timestamp", time.time()))
        
        html += f"""
        <div style='margin-bottom: 15px;'>
            <div style='font-weight: bold; color: #444;'>{speaker}</div>
            <div style='margin: 5px 0;'>{text}</div>
            <div style='font-size: 0.8em; color: #666;'>{timestamp}</div>
        </div>
        """
    html += "</div>"
    return html

def load_config(config_path: str) -> Dict:
    """
    Load configuration from a JSON or YAML file
    :param config_path: Path to configuration file
    :return: Configuration dictionary
    """
    try:
        if not os.path.exists(config_path):
            return {}
            
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                import yaml
                return yaml.safe_load(f)
            else:
                raise ValueError("Unsupported configuration file format")
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return {}

def save_config(config: Dict, config_path: str) -> bool:
    """
    Save configuration to a file
    :param config: Configuration dictionary
    :param config_path: Path to save configuration
    :return: Success status
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.endswith('.json'):
                json.dump(config, f, indent=2)
            elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                import yaml
                yaml.dump(config, f)
            else:
                raise ValueError("Unsupported configuration file format")
        return True
    except Exception as e:
        print(f"Error saving configuration: {str(e)}")
        return False

def create_error_response(error: str, details: Optional[Dict] = None) -> Dict:
    """
    Create a standardized error response
    :param error: Error message
    :param details: Additional error details
    :return: Error response dictionary
    """
    response = {
        'status': 'error',
        'message': error,
        'timestamp': datetime.now().isoformat()
    }
    
    if details:
        response['details'] = details
        
    return response

def validate_api_keys() -> Dict[str, bool]:
    """
    Validate the presence of required API keys
    :return: Dictionary of API key validation results
    """
    required_keys = {
        'OPENAI_API_KEY': 'OpenAI API key',
        'ELEVENLABS_API_KEY': 'ElevenLabs API key',
        'TAVILY_API_KEY': 'Tavily API key'
    }
    
    validation_results = {}
    for env_var, description in required_keys.items():
        key_value = os.getenv(env_var)
        validation_results[description] = bool(key_value)
        
    return validation_results

def generate_session_id() -> str:
    """
    Generate a unique session identifier
    :return: Session ID string
    """
    timestamp = int(time.time() * 1000)
    random_suffix = os.urandom(4).hex()
    return f"session_{timestamp}_{random_suffix}"
