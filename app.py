import streamlit as st
import os
from dotenv import load_dotenv
import yaml
import asyncio
from typing import Dict, Optional
import time
import nest_asyncio
import warnings
import queue
import threading

# Suppress FP16 warning from Whisper
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Enable nested event loops
nest_asyncio.apply()

# Import custom modules
from modules.speech.listener import AudioListener
from modules.speech.speaker import Speaker
from modules.memory.vectordb import VectorStorage
from modules.memory.context import ContextManager
from modules.ai.persona import VirtualTeamMember
from modules.ai.thinking import ThoughtEngine
from modules.ai.search import SearchEngine
from modules.utils.helpers import (
    format_timestamp,
    format_references,
    format_conversation,
    load_config,
    validate_api_keys,
    generate_session_id
)
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# Initialize asyncio loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

class VirtualTeamMemberApp:
    def __init__(self):
        """Initialize the Virtual Team Member application"""
        print("Initializing VirtualTeamMemberApp...")
        # Load configuration
        self.config = load_config('config/config.yaml')
        
        # Initialize components
        self.setup_components()
        
        # Initialize session state
        self.init_session_state()
        print("VirtualTeamMemberApp initialized successfully")
        
    def setup_components(self):
        """Set up all application components"""
        print("Setting up components...")
        # Initialize memory components
        self.vector_storage = VectorStorage(
            persist_directory=self.config['memory']['vector_db']['persist_directory']
        )
        self.context_manager = ContextManager(self.vector_storage)
        print("Memory components initialized")
        
        # Initialize AI components
        self.virtual_member = VirtualTeamMember(
            name=self.config['ai']['persona']['name'],
            role=self.config['ai']['persona']['role'],
            personality_traits=self.config['ai']['persona']['personality_traits']
        )
        self.thought_engine = ThoughtEngine()
        self.search_engine = SearchEngine()
        print("AI components initialized")
        
        # Initialize speech components
        print("Initializing speech components...")
        self.speaker = Speaker()
        self.listener = AudioListener(
            callback=self.handle_speech_input,
            language=self.config['speech']['language_codes']['English'],
            sample_rate=self.config['speech']['sample_rate']
        )
        print("Speech components initialized")

    def init_session_state(self):
        """Initialize Streamlit session state variables"""
        print("Initializing session state...")
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'is_listening' not in st.session_state:
            st.session_state.is_listening = False
        if 'selected_language' not in st.session_state:
            st.session_state.selected_language = 'English'
        if 'session_id' not in st.session_state:
            st.session_state.session_id = generate_session_id()
        if 'references' not in st.session_state:
            st.session_state.references = []
        if 'error_message' not in st.session_state:
            st.session_state.error_message = None
        if 'last_transcript' not in st.session_state:
            st.session_state.last_transcript = ""
        if 'last_update' not in st.session_state:
            st.session_state.last_update = time.time()
        print("Session state initialized")

    def handle_speech_input(self, text: str):
        """Handle speech input in the main thread"""
        try:
            print(f"Handling speech input: {text}")
            # Update transcript immediately
            st.session_state.last_transcript = text
            st.session_state.last_update = time.time()
            
            # Add to conversation history
            message_id = self.context_manager.add_message(
                text=text,
                speaker="Human",
                metadata={"timestamp": time.time()}
            )
            
            # Update UI with new transcript
            st.session_state.conversation_history = (
                self.context_manager.get_conversation_summary()
            )
            print("Conversation history updated")
            
            # Run async processing in the event loop
            print("Starting async processing...")
            loop.create_task(self.async_process_input(text))
            
        except Exception as e:
            error_msg = f"Error handling speech input: {str(e)}"
            print(error_msg)
            st.session_state.error_message = error_msg

    async def async_process_input(self, text: str):
        """
        Asynchronously process the input and generate response
        :param text: Input text to process
        """
        try:
            print(f"Processing input asynchronously: {text}")
            # Get context and process thought
            context = self.context_manager.get_relevant_context(text)
            response, references = await self.thought_engine.process_thought(
                context=context,
                current_discussion=text
            )
            print("Thought processing complete")
            
            # Add virtual member's response
            self.context_manager.add_message(
                text=response,
                speaker=self.config['ai']['persona']['name'],
                metadata={"timestamp": time.time()}
            )
            print("Response added to context")
            
            # Add references
            for ref in references:
                self.context_manager.add_reference(
                    content=ref['snippet'],
                    url=ref['url'],
                    metadata={"title": ref['title']}
                )
            print("References added")
            
            # Update UI
            st.session_state.conversation_history = (
                self.context_manager.get_conversation_summary()
            )
            st.session_state.references = self.context_manager.get_active_references()
            print("UI updated with new content")
            
            # Speak response (now synchronous)
            print("Speaking response...")
            self.speaker.speak(response)
            print("Response spoken")
            
        except Exception as e:
            error_msg = f"Error processing input: {str(e)}"
            print(error_msg)
            st.session_state.error_message = error_msg

    def render_ui(self):
        """Render the Streamlit UI"""
        # Page configuration
        st.set_page_config(
            page_title="Virtual Team Member",
            page_icon="🤖",
            layout="wide"
        )
        
        # Title and description
        st.title("Virtual Team Member Assistant")
        
        # Display any error messages
        if st.session_state.error_message:
            st.error(st.session_state.error_message)
            st.session_state.error_message = None
        
        # Sidebar configuration
        self.render_sidebar()
        
        # Main content
        self.render_main_content()
        
        # Auto-rerun if listening
        if st.session_state.is_listening:
            time.sleep(0.1)  # Small delay to prevent too frequent reruns
            st.rerun()

    def render_sidebar(self):
        """Render the sidebar content"""
        with st.sidebar:
            # Language selector
            st.selectbox(
                "Select Language",
                options=list(self.config['speech']['language_codes'].keys()),
                key='selected_language',
                on_change=self.update_language
            )
            
            # API key status
            st.subheader("API Key Status")
            key_status = validate_api_keys()
            for key, status in key_status.items():
                st.write(f"{key}: {'✅' if status else '❌'}")
            
            # Session information
            st.subheader("Session Information")
            st.write(f"Session ID: {st.session_state.session_id}")
            st.write(f"Messages: {len(st.session_state.conversation_history)}")
            st.write(f"Last Update: {time.strftime('%H:%M:%S', time.localtime(st.session_state.last_update))}")

    def render_main_content(self):
        """Render the main content area"""
        # Control buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(
                "Start Listening",
                type="primary",
                disabled=st.session_state.is_listening
            ):
                try:
                    print("Start Listening button clicked")
                    self.start_listening()
                except Exception as e:
                    error_msg = str(e)
                    print(f"Error starting listening: {error_msg}")
                    st.session_state.error_message = error_msg
                    st.session_state.is_listening = False
                    st.rerun()
        
        with col2:
            if st.button(
                "Stop",
                type="secondary",
                disabled=not st.session_state.is_listening
            ):
                print("Stop button clicked")
                self.stop_listening()
        
        with col3:
            if st.button("Clear History"):
                print("Clear History button clicked")
                self.clear_history()
        
        # Live transcript area
        st.markdown("### Live Transcript")
        if st.session_state.is_listening:
            st.info(f"🎤 {st.session_state.last_transcript}")
        
        # Content columns
        conversation_col, references_col = st.columns([3, 1])
        
        with conversation_col:
            st.subheader("Conversation History")
            st.markdown(
                format_conversation(st.session_state.conversation_history),
                unsafe_allow_html=True
            )
        
        with references_col:
            st.subheader("References")
            st.markdown(
                format_references(st.session_state.references),
                unsafe_allow_html=True
            )

    def start_listening(self):
        """Start the listening process"""
        if not st.session_state.is_listening:
            try:
                print("Starting listening process...")
                st.session_state.is_listening = True
                self.listener.set_language(
                    self.config['speech']['language_codes'][st.session_state.selected_language]
                )
                self.listener.start()
                self.speaker.start()
                print("Listening process started successfully")
            except Exception as e:
                error_msg = f"Could not start audio: {str(e)}. Please check microphone permissions."
                print(error_msg)
                st.session_state.is_listening = False
                raise RuntimeError(error_msg)

    def stop_listening(self):
        """Stop the listening process"""
        if st.session_state.is_listening:
            print("Stopping listening process...")
            st.session_state.is_listening = False
            self.listener.stop()
            self.speaker.stop()
            print("Listening process stopped")

    def clear_history(self):
        """Clear conversation history"""
        print("Clearing conversation history...")
        self.context_manager.clear_context()
        st.session_state.conversation_history = []
        st.session_state.references = []
        st.session_state.last_transcript = ""
        print("Conversation history cleared")

    def update_language(self):
        """Update the speech recognition language"""
        if st.session_state.is_listening:
            print(f"Updating language to: {st.session_state.selected_language}")
            self.listener.set_language(
                self.config['speech']['language_codes'][st.session_state.selected_language]
            )
            print("Language updated")

def main():
    """Main application entry point"""
    print("Starting application...")
    # Load environment variables
    load_dotenv()
    
    # Create and run application
    app = VirtualTeamMemberApp()
    app.render_ui()

if __name__ == "__main__":
    main()
