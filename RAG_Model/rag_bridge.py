from Retrieval import qa_chain, create_vectorstore 

RAG_THRESHOLD = 0.60

_vectorstore = create_vectorstore()
_chain = qa_chain(vector_store=_vectorstore)

_NO_ANSWER_PHRASE = "I don't known I'm only trained on financial data"

def get_rag_answer(query: str) -> str:
    """
    Main entry pont for orchestration.py.
    Returns (answer, score).
    Returns (None, 0.0) if there's nothing relevant, or if the LLM
    itself admits it doesn't known - either signal means 'fall back to MCP'.
    'sources' is a list of source labels pulled from each retrieved documents' 
    metadata such as page number and name of that pdf, for display in the UI.
    """

    scored_results = _vectorstore.similarity_search_with_score(query, k = 3)
    if not scored_results:
        return None, 0.0, [] 

    top_score = scored_results[0][1]

    response = _chain.invoke({'query' : query})
    answer = response['result']

    if _NO_ANSWER_PHRASE in answer.lower():
        return None, 0.0, []

    sources = []
    for doc in response.get('source_documents', []):
        label = doc.metadata.get('source') or doc.metadata.get('file_name') or "Financial document"

        if label not in sources:
            sources.append(label) 

    return answer, top_score, sources


if __name__ == "__main__":
    test_query = "What is a Balance Sheet?"
    answer, score, sources = get_rag_answer(test_query)
    print("Score:- ", score)
    print("Answer:- ", answer) 
    print("Sources:- ", sources)



