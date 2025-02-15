from typing import List, Dict, Optional
from tavily import TavilyClient
import os
import asyncio
from datetime import datetime

class SearchEngine:
    def __init__(self):
        """Initialize the search engine with Tavily API"""
        self.api_key = os.getenv('TAVILY_API_KEY')
        if not self.api_key:
            raise ValueError("Tavily API key not found in environment variables")
        
        self.client = TavilyClient(api_key=self.api_key)
        self.search_history = []

    async def search(self, 
                    query: str, 
                    search_depth: str = "advanced",
                    max_results: int = 5,
                    include_domains: Optional[List[str]] = None,
                    exclude_domains: Optional[List[str]] = None) -> Dict:
        """
        Perform an internet search using Tavily API
        :param query: Search query
        :param search_depth: Search depth (basic or advanced)
        :param max_results: Maximum number of results to return
        :param include_domains: List of domains to include in search
        :param exclude_domains: List of domains to exclude from search
        :return: Search results and metadata
        """
        try:
            # Prepare search parameters
            search_params = {
                "query": query,
                "search_depth": search_depth,
                "max_results": max_results
            }
            
            if include_domains:
                search_params["include_domains"] = include_domains
            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains

            # Perform search
            results = self.client.search(**search_params)
            
            # Format and store search history
            search_record = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'parameters': search_params,
                'results_count': len(results) if results else 0
            }
            self.search_history.append(search_record)
            
            return {
                'status': 'success',
                'results': results,
                'metadata': search_record
            }
            
        except Exception as e:
            error_response = {
                'status': 'error',
                'error': str(e),
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
            self.search_history.append(error_response)
            return error_response

    async def analyze_results(self, results: Dict) -> Dict:
        """
        Analyze search results to extract key information
        :param results: Search results from Tavily
        :return: Analysis of the results
        """
        if results.get('status') != 'success':
            return {
                'status': 'error',
                'error': 'Invalid search results provided'
            }

        try:
            analysis = {
                'total_results': len(results['results']),
                'domains': set(),
                'topics': [],
                'summary': []
            }

            for result in results['results']:
                # Collect unique domains
                if 'url' in result:
                    domain = result['url'].split('/')[2] if len(result['url'].split('/')) > 2 else result['url']
                    analysis['domains'].add(domain)

                # Collect titles and snippets for topic analysis
                if 'title' in result:
                    analysis['topics'].append(result['title'])
                if 'snippet' in result:
                    analysis['summary'].append(result['snippet'])

            return {
                'status': 'success',
                'analysis': {
                    'total_results': analysis['total_results'],
                    'unique_domains': list(analysis['domains']),
                    'key_topics': analysis['topics'],
                    'summary_points': analysis['summary']
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': f'Error analyzing results: {str(e)}'
            }

    def get_search_history(self, 
                          limit: Optional[int] = None, 
                          query_filter: Optional[str] = None) -> List[Dict]:
        """
        Get search history with optional filtering
        :param limit: Maximum number of history items to return
        :param query_filter: Filter history by query content
        :return: List of search history items
        """
        history = self.search_history
        
        if query_filter:
            history = [
                item for item in history 
                if query_filter.lower() in item.get('query', '').lower()
            ]
            
        if limit:
            history = history[-limit:]
            
        return history

    def clear_history(self):
        """Clear the search history"""
        self.search_history = []

    async def batch_search(self, 
                          queries: List[str], 
                          **search_params) -> List[Dict]:
        """
        Perform multiple searches in parallel
        :param queries: List of search queries
        :param search_params: Additional search parameters
        :return: List of search results
        """
        tasks = [
            self.search(query, **search_params)
            for query in queries
        ]
        
        results = await asyncio.gather(*tasks)
        return results
