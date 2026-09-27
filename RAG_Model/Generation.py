from Retrieval import qa_chain

user_input = str(input("Enter you're query here..."))


response = qa_chain().invoke({'query':user_input}) 
print("Answer:- ", response['result'])
print("Source Documents:- ", response['source_documents'])


