import os 
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer
from tavily import TavilyClient


load_dotenv()

tavily_client = TavilyClient(api_key= os.getenv("TAVILY_API_KEY"))

mcp = MCPServer('web-search-server')

@mcp.tool()
def web_search(query: str) -> str:
    """ 
    Search the web for current information using Tavily.
    Use this when the RAG knowledge base does not contain 
    an answer to the user's financial question. 
    """

    response  = tavily_client.search(
        query= query,
        max_results= 5,
        search_depth= 'basic',
    )

    results = response.get('results', [])

    if not results:
        return "No result found."

    format = "\n\n".join(
        f"Title: {r['title']}\n URL: {r['url']}\n Snippet: {r['content']}"
        
        for r in results
    )

    return format


if __name__ == "__main__":

    mcp.run() 

