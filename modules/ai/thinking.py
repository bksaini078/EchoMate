from typing import Dict, List, Optional, Tuple
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferMemory
from langchain_openai import AzureChatOpenAI
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from tavily import TavilyClient
import os
import json
from crewai import Agent, Task, Crew, Process

class ThoughtEngine:
    def __init__(self):
        """Initialize the thought engine with necessary components"""
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        
        if not self.tavily_api_key:
            raise ValueError("Tavily API key not found in environment variables")
        
        self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
        self.llm = AzureChatOpenAI(
            model=os.getenv('AZURE_MODEL_NAME', "gpt-4-32k-0613"),
            api_version=os.getenv('AZURE_API_VERSION', "2024-02-01"),
            api_key=os.getenv('AZURE_API_KEY'),
            azure_endpoint=os.getenv('AZURE_ENDPOINT'),
            temperature=0.7
        )
        
        # Initialize tools and agents
        self._init_tools()
        self._init_agents()

    def _init_tools(self):
        """Initialize available tools"""
        self.tools = [
            Tool(
                name="Internet Search",
                func=self._search_internet,
                description="Search the internet for information using Tavily API"
            ),
            Tool(
                name="Analyze Context",
                func=self._analyze_context,
                description="Analyze the current conversation context"
            ),
            Tool(
                name="Generate Response",
                func=self._generate_response,
                description="Generate a response based on analysis"
            )
        ]

    def _init_agents(self):
        """Initialize specialized agents using CrewAI"""
        # Research Agent
        self.researcher = Agent(
            role='Research Specialist',
            goal='Find and validate relevant information',
            backstory='Expert at finding and analyzing information from various sources',
            tools=[self.tools[0]],  # Internet Search tool
            llm=self.llm
        )
        
        # Analysis Agent
        self.analyst = Agent(
            role='Context Analyst',
            goal='Analyze conversation context and identify key points',
            backstory='Specialist in understanding context and extracting insights',
            tools=[self.tools[1]],  # Analyze Context tool
            llm=self.llm
        )
        
        # Response Agent
        self.responder = Agent(
            role='Communication Specialist',
            goal='Generate appropriate and engaging responses',
            backstory='Expert in crafting contextually appropriate responses',
            tools=[self.tools[2]],  # Generate Response tool
            llm=self.llm
        )

    async def _search_internet(self, query: str) -> Dict:
        """
        Search the internet using Tavily API
        :param query: Search query
        :return: Search results
        """
        try:
            search_result = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=5
            )
            return {
                'results': search_result,
                'status': 'success'
            }
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }

    async def _analyze_context(self, 
                             context: Dict,
                             current_discussion: str) -> Dict:
        """
        Analyze the current context
        :param context: Current context dictionary
        :param current_discussion: Current discussion text
        :return: Analysis results
        """
        try:
            # Create analysis crew
            analysis_crew = Crew(
                agents=[self.analyst],
                tasks=[
                    Task(
                        description=f"""
                        Analyze the following context and discussion:
                        Context: {json.dumps(context)}
                        Discussion: {current_discussion}
                        
                        Provide:
                        1. Key topics and themes
                        2. Participant dynamics
                        3. Important points raised
                        4. Areas needing clarification
                        """,
                        agent=self.analyst
                    )
                ],
                process=Process.sequential
            )
            
            result = await analysis_crew.run()
            return {
                'analysis': result,
                'status': 'success'
            }
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }

    async def _generate_response(self, 
                               context: Dict,
                               analysis: Dict,
                               search_results: Optional[Dict] = None) -> str:
        """
        Generate a response based on analysis and search results
        :param context: Current context
        :param analysis: Analysis results
        :param search_results: Optional search results
        :return: Generated response
        """
        try:
            # Create response crew
            response_crew = Crew(
                agents=[self.responder],
                tasks=[
                    Task(
                        description=f"""
                        Generate a response based on:
                        Context: {json.dumps(context)}
                        Analysis: {json.dumps(analysis)}
                        Search Results: {json.dumps(search_results) if search_results else 'None'}
                        
                        Ensure the response is:
                        1. Relevant to the current discussion
                        2. Incorporates any relevant search results
                        3. Maintains appropriate tone and style
                        4. Adds value to the conversation
                        """,
                        agent=self.responder
                    )
                ],
                process=Process.sequential
            )
            
            result = await response_crew.run()
            return result
        except Exception as e:
            return f"Error generating response: {str(e)}"

    async def process_thought(self, 
                            context: Dict,
                            current_discussion: str,
                            query: Optional[str] = None) -> Tuple[str, List[Dict]]:
        """
        Process a complete thought cycle
        :param context: Current context
        :param current_discussion: Current discussion text
        :param query: Optional specific query to address
        :return: Tuple of (response, references)
        """
        try:
            # 1. Analyze context
            analysis = await self._analyze_context(context, current_discussion)
            
            # 2. Determine if internet search is needed
            search_results = None
            if query or (analysis.get('analysis', {}).get('areas_needing_clarification')):
                search_query = query or analysis['analysis']['areas_needing_clarification'][0]
                search_results = await self._search_internet(search_query)
            
            # 3. Generate response
            response = await self._generate_response(
                context,
                analysis,
                search_results
            )
            
            # 4. Extract references from search results
            references = []
            if search_results and search_results.get('status') == 'success':
                references = [
                    {
                        'title': result.get('title', ''),
                        'url': result.get('url', ''),
                        'snippet': result.get('snippet', '')
                    }
                    for result in search_results.get('results', [])
                ]
            
            return response, references
            
        except Exception as e:
            return f"Error in thought process: {str(e)}", []

    def update_configuration(self, 
                           new_temperature: Optional[float] = None,
                           new_model: Optional[str] = None):
        """
        Update the configuration of the thought engine
        :param new_temperature: New temperature value for LLM
        :param new_model: New model name for LLM
        """
        if new_temperature is not None:
            self.llm.temperature = new_temperature
        if new_model is not None:
            self.llm.model_name = new_model
