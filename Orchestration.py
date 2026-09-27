import asyncio
import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai 
from google.genai import types
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "RAG_Model"))
from rag_bridge import get_rag_answer, RAG_THRESHOLD # type: ignore

load_dotenv() 

# Gemini LLM 
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# connect mcp_server.py file 
server_params = StdioServerParameters(
    command='python',
    args=['MCP layer/mcp_server.py'],
    env=os.environ.copy(),
)

# The Gemini LLM does't not understand MCP server format directly it only understands its own format
#  so that's why I explicitly defind "FunctionDeclaration" format for Gemini LLM .
def mcp_tool_to_gemini_declaration(mcp_tool) -> types.FunctionDeclaration:
    """Convert an MCP tool's schema into a Gemini FunctionDeclaration."""
    return types.FunctionDeclaration(
        name = mcp_tool.name,
        description= mcp_tool.description or "",
        parameters= mcp_tool.input_schema,
    )


async def ask_mcp_fallback(query: str) -> str:
    """
    Called when a RAG has no answer. Spawns the MCP server, gives Gemini acces to its web_search tool,
    and lets Gemini decide how to use it.
    """
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. Discover tools exposed by mcp_server.py
            mcp_tools = (await session.list_tools()).tools
            gemini_tool = types.Tool(
                function_declarations=[
                    mcp_tool_to_gemini_declaration(t) for t in mcp_tools
                ]
            )
            config = types.GenerateContentConfig(
                tools=[gemini_tool],
                system_instruction="Answer in 5-6 lines maximum. Be concise and direct.",
                )

            # 2. Give Gemini the tool and the user's question
            chat = client.chats.create(model = 'gemini-3.5-flash-lite', config = config)
            response = chat.send_message(query)

            # 3. If Gemini wants to call web_search, run it via MCP and
            # send the result back to Gemini can write the final answer
            part = response.candidates[0].content.parts[0]
            while part.function_call and part.function_call.name:
                fn_call = part.function_call
                tool_result = await session.call_tool(
                    fn_call.name, dict(fn_call.args)
                )
                result_text = '\n'.join(
                    block.text for block in tool_result.content
                )

                response = chat.send_message(
                    types.Part.from_function_response(
                        name = fn_call.name,
                        response={'result': result_text},
                    )
                )
                part = response.candidates[0].content.parts[0]

            # Gemini answered directly without needing the tool 
            return response.text


def route_query(query: str) -> str:
    """
    Main entry point for the app.
    Tries RAG first; falls back to MCP if RAG's score is below
    threshold or if the LLM admits it doesn't know.
    Returns {'answer': str, 'source': 'RAG' or 'MCP', 'sources': list}
    """

    # Generate response with RAG Model 
    rag_answer, rag_score, sources = get_rag_answer(query)
    if rag_answer and rag_score >= RAG_THRESHOLD:
        return {'answer':rag_answer, 'source': 'RAG', 'sources': sources}

    # Generate response with MCP server
    mcp_answer = asyncio.run(ask_mcp_fallback(query))
    return {'answer': mcp_answer, 'source': 'MCP', 'sources': ["Web-search with Tavily"]}


if __name__ == '__main__':
    # test_query = "What is today's RBI repo rate?"
    test_query = 'What is Balance Sheet?'
    result = route_query(test_query)
    print("Source:- ", result['source'])
    print(result['answer'])
    print("Source Documents:- ", result['sources'])

