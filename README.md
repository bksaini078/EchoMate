# Virtual Team Member Assistant

A virtual team member that participates in physical meetings by listening to conversations, understanding context, and providing valuable input through speech interaction.

## Features

- Real-time speech recognition and transcription
- Natural text-to-speech responses
- Context-aware conversation understanding
- Internet search capabilities for fact-checking and references
- Vector database for conversation memory
- Multi-language support
- Simple and intuitive UI

## Prerequisites

- Python 3.8 or higher
- Required API keys and configurations:
  - Azure OpenAI API key and endpoint
  - ElevenLabs API key (for text-to-speech)
  - Tavily API key (for internet search)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd TeamSupport
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:
```env
AZURE_API_KEY=your_azure_api_key
AZURE_ENDPOINT=your_azure_endpoint
AZURE_MODEL_NAME=gpt-4-32k-0613
AZURE_API_VERSION=2024-02-01
ELEVENLABS_API_KEY=your_elevenlabs_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Configuration

The application can be configured through the `config/config.yaml` file. Key configuration options include:

- Speech recognition settings
- Text-to-speech parameters
- AI model configuration
- Memory and context settings
- UI customization
- Logging preferences

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. The application will open in your default web browser.

3. Select your preferred language from the sidebar.

4. Click "Start Listening" to begin a session.

5. The virtual team member will:
   - Listen to the conversation
   - Transcribe speech in real-time
   - Process and understand the context
   - Provide relevant responses
   - Search for additional information when needed
   - Store conversation history for future reference

## Project Structure

```
TeamSupport/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── README.md             # Project documentation
├── config/
│   └── config.yaml       # Configuration settings
├── modules/
│   ├── speech/
│   │   ├── listener.py   # Speech recognition
│   │   └── speaker.py    # Text-to-speech
│   ├── memory/
│   │   ├── vectordb.py   # Vector database operations
│   │   └── context.py    # Context management
│   ├── ai/
│   │   ├── persona.py    # Virtual member personality
│   │   ├── thinking.py   # Decision making & response
│   │   └── search.py     # Internet search capabilities
│   └── utils/
│       └── helpers.py    # Utility functions
└── assets/
    └── animations/       # UI animation files
```

## Key Components

### Speech Processing
- Uses OpenAI Whisper for speech recognition
- ElevenLabs for natural text-to-speech
- Real-time audio processing and transcription

### AI Brain
- GPT-4 for natural language understanding
- CrewAI for role-based interactions
- Custom prompting system for personality maintenance
- Tavily integration for internet search

### Memory System
- ChromaDB for vector storage
- Context-aware conversation management
- Reference tracking and retrieval

### User Interface
- Clean, minimal Streamlit interface
- Real-time conversation display
- Audio visualization
- Reference management

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Azure OpenAI for GPT-4 model
- OpenAI Whisper
- ElevenLabs for text-to-speech
- Tavily for search capabilities
- Streamlit for the UI framework
