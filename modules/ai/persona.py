from typing import Dict, List, Optional
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
import os

class VirtualTeamMember:
    def __init__(self, 
                 name: str = "Theo",
                 role: str = "Technical Advisor",
                 personality_traits: List[str] = None):
        """
        Initialize the virtual team member with personality traits
        :param name: Name of the virtual team member
        :param role: Role in the team
        :param personality_traits: List of personality traits
        """
        self.name = name
        self.role = role
        self.personality_traits = personality_traits or [
            "analytical",
            "supportive",
            "curious",
            "professional"
        ]
        
        # Initialize Azure OpenAI LLM
        self.llm = AzureChatOpenAI(
            model=os.getenv('AZURE_MODEL_NAME', "gpt-4-32k-0613"),
            api_version=os.getenv('AZURE_API_VERSION', "2024-02-01"),
            api_key=os.getenv('AZURE_API_KEY'),
            azure_endpoint=os.getenv('AZURE_ENDPOINT'),
            temperature=0.7
        )
        
        # Initialize prompt templates
        self._init_prompts()

    def _init_prompts(self):
        """Initialize various prompt templates for different interaction types"""
        self.response_prompt = PromptTemplate(
            input_variables=["name", "role", "traits", "context", "current_discussion", "query"],
            template="""
            You are {name}, a {role} with the following personality traits: {traits}.
            
            Current Context:
            {context}
            
            Current Discussion:
            {current_discussion}
            
            Based on the context and your role, provide a response to: {query}
            
            Remember to:
            1. Stay in character and maintain consistency with your personality traits
            2. Be concise but informative
            3. Show emotional intelligence and read the room
            4. Provide constructive input that moves the discussion forward
            """
        )
        
        self.interjection_prompt = PromptTemplate(
            input_variables=["name", "role", "traits", "context", "current_discussion"],
            template="""
            You are {name}, a {role} with the following personality traits: {traits}.
            
            Current Context:
            {context}
            
            Current Discussion:
            {current_discussion}
            
            Decide if you should interject in the conversation. If yes, provide your input.
            Consider:
            1. Is your input valuable to the current discussion?
            2. Is this an appropriate time to speak?
            3. Will your contribution move the conversation forward?
            
            If you decide to interject, provide your response. If not, respond with "LISTEN".
            """
        )
        
        self.reflection_prompt = PromptTemplate(
            input_variables=["name", "role", "traits", "context", "discussion_summary"],
            template="""
            You are {name}, a {role} with the following personality traits: {traits}.
            
            Context of the Discussion:
            {context}
            
            Discussion Summary:
            {discussion_summary}
            
            Reflect on the discussion and provide:
            1. Key insights gained
            2. Potential areas for further exploration
            3. Any concerns or risks identified
            4. Suggested next steps
            """
        )

    async def generate_response(self, 
                              context: Dict, 
                              current_discussion: str, 
                              query: str) -> str:
        """
        Generate a response based on context and current discussion
        :param context: Current context dictionary
        :param current_discussion: Current discussion text
        :param query: Specific query to respond to
        :return: Generated response
        """
        chain = LLMChain(llm=self.llm, prompt=self.response_prompt)
        response = await chain.arun(
            name=self.name,
            role=self.role,
            traits=", ".join(self.personality_traits),
            context=str(context),
            current_discussion=current_discussion,
            query=query
        )
        return response.strip()

    async def should_interject(self, 
                             context: Dict, 
                             current_discussion: str) -> Optional[str]:
        """
        Decide whether to interject in the conversation
        :param context: Current context dictionary
        :param current_discussion: Current discussion text
        :return: Interjection text if should interject, None otherwise
        """
        chain = LLMChain(llm=self.llm, prompt=self.interjection_prompt)
        response = await chain.arun(
            name=self.name,
            role=self.role,
            traits=", ".join(self.personality_traits),
            context=str(context),
            current_discussion=current_discussion
        )
        
        response = response.strip()
        return None if response == "LISTEN" else response

    async def reflect_on_discussion(self, 
                                  context: Dict, 
                                  discussion_summary: str) -> str:
        """
        Reflect on the discussion and provide insights
        :param context: Context dictionary
        :param discussion_summary: Summary of the discussion
        :return: Reflection and insights
        """
        chain = LLMChain(llm=self.llm, prompt=self.reflection_prompt)
        response = await chain.arun(
            name=self.name,
            role=self.role,
            traits=", ".join(self.personality_traits),
            context=str(context),
            discussion_summary=discussion_summary
        )
        return response.strip()

    def update_personality(self, 
                         new_traits: List[str], 
                         new_name: Optional[str] = None,
                         new_role: Optional[str] = None):
        """
        Update the virtual team member's personality
        :param new_traits: New personality traits
        :param new_name: New name (optional)
        :param new_role: New role (optional)
        """
        self.personality_traits = new_traits
        if new_name:
            self.name = new_name
        if new_role:
            self.role = new_role
