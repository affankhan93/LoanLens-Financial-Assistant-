# LoanLens

## 1. Overview

LoanLens is a financial assistant app built with Streamlit. It has two main jobs:

1. **Check if a loan is likely to be approved or not**, using a machine learning model, and show a confidence score along with a few key numbers (total income, monthly installment, collateral-to-loan ratio).
2. **Answer financial questions** through a chatbot. The chatbot first looks in a knowledge base built from real financial documents. If it doesn't find the answer there, it automatically searches the web instead, so the user still gets an answer.

The goal is to give someone a quick, honest read on their loan eligibility and a reliable place to ask financial questions, without needing to dig through documents or search the web themselves.

## 2. Architecture

The app is built around one central piece of logic called the **Orchestration Logic**. It decides what happens after a user asks something.

**Flow:**
```
User
  |
  v
Orchestration Logic
  |
  |--> ML Tool (Loan Approval System) 
  |
  |--> RAG Tool (Financial PDFs) (Groq LLM)
  |
  |--> MCP Layer (External Tools)
        |
        v
     Web Search Tool (Tavily)
        |
        v
     Gemini LLM
  |
  v
Response Generation + Sources
  |
  v
User Interface (UI)
```

**Note:- To know more about the actual architecture flow and roadmap. I have created Arcitecture folder you can refer that.** 
 

**How a question is handled, step by step:**

1. If the user is checking loan eligibility, the request goes straight to the **ML Tool**.
2. If the user asks a financial question, it first goes to the **RAG Tool**, which searches the financial PDF knowledge base.
3. The RAG Tool returns a similarity score. If that score is high enough (above a set threshold), the RAG answer is used, along with the source document it came from.
4. If the score is too low, or the RAG system itself says it doesn't know the answer, the **MCP Layer** takes over instead.
5. The MCP Layer uses a **web search tool (Tavily)** to find current information, then **Gemini** writes the final answer from those search results.
6. Whichever tool answered, the final response (and its source, if there is one) is sent back and shown in the **User Interface**.

This means the app always tries the most reliable, curated source first, and only reaches out to the open web when it genuinely needs to.

## 3. Project Structure

```
LoanLens/
├── app.py                     # Main Streamlit app: auth, UI, loan form, chatbot
├── Orchestration.py            # Decides RAG vs MCP, and runs the MCP fallback
├── requirements.txt            # All Python dependencies
├── .env                        # API keys and secrets (I have not commit this file).
│
├── ML_Model/
│   ├── model.ipynb              # Notebook: data cleaning, feature engineering, training and Evaluate.
│   └── model_artifacts.pkl      # Saved model, scaler, and column/category info
│
├── RAG_Model/
│   ├── Ingestion.py             # Loads financial PDFs, creates embeddings, stores them in Pinecone 
│   ├── Retrieval.py             # Connects to Pinecone, builds the question-answering chain
│   ├── Generation.py            # Runs a single question through the RAG chain (manual testing)
│   ├── Ragas_Evalution.py       # Evaluates the RAG pipeline using RAGAS metrics 
│   └── rag_bridge.py            # Connects the RAG pipeline to Orchestration.py, adds scoring
│
├── MCP layer/
│   └── mcp_server.py            # MCP server exposing the Tavily web search tool 
│
└── Supa_database/
    └── Authentication.py        # Username/password signup and login, using Supabase
```

## 4. How Each Tool Works

**Authentication (Supabase)**
The app stores usernames and passwords in a Supabase database table. Passwords are never stored as plain text — they are hashed with bcrypt before being saved. When someone logs in, their entered password is hashed and compared against the stored hash.

**ML Tool (Loan Approval System)**
A supervised machine learning model (Random Forest) trained on loan application data. It takes in details like income, credit score, loan amount, and employment status, applies the same feature engineering used during training (total income, monthly installment, collateral-to-loan ratio), and predicts whether the loan is likely to be approved, along with a confidence percentage.

**RAG Tool (Retrieval Augmented Generation)**
Financial PDF documents are split into chunks and converted into embeddings using a HuggingFace model. These embeddings are stored in a Pinecone vector database. When a question comes in, the most relevant chunks are retrieved, and a Groq-hosted LLM generates an answer using only that retrieved content, guided by a custom prompt that tells it not to make things up.

**MCP Layer (Model Context Protocol)**
This is the fallback system. It only activates when the RAG Tool doesn't have a good enough answer. It runs a small MCP server that exposes a web search tool (Tavily). Gemini, the LLM used here, decides on its own whether it needs to call that search tool, then writes the final answer using whatever it finds.

**Orchestration Logic**
This is the decision-maker that ties everything together. For loan questions, it goes straight to the ML Tool. For general questions, it checks the RAG Tool's confidence score against a set threshold — high enough, and the RAG answer is used; too low, and it hands the question to the MCP Layer instead.

**User Interface (Streamlit)**
The UI has a login/signup page, a loan eligibility form, and a floating chatbot button. The loan eligibility results are shown with a confidence score and key financial metrics. Chatbot answers show which system generated them (RAG or MCP), and RAG answers additionally show the source document they came from.

## 5. Evaluation Result

The ML model was evaluated on a held-out test set:

| Metric | Score |
|---|---|
| Accuracy | 88% |
| Precision | 83.63% |

This means the model correctly predicts loan approval outcomes about 88% of the time, and when it predicts a loan will be approved, it's right about 83.63% of the time.

The RAG pipeline was separately evaluated using RAGAS metrics (results saved in `ragas_financial_result.csv`), covering how faithful its answers are to the source documents and how relevant they are to the questions asked.

## 6. Features

- Check loan eligibility instantly, with a clear approve/not-approve result and a confidence score
- See the key numbers behind the decision: total income, monthly installment, and collateral-to-loan ratio
- Ask any financial question through a chatbot
- Chatbot answers show whether they came from the curated knowledge base or a live web search
- Source documents are shown for knowledge-base answers, so you know where the information came from
- Secure login and signup, with passwords stored safely (hashed, never in plain text)
- Clean, easy-to-read interface with a consistent color theme

## 7. Tech Stack

**Machine Learning**
scikit-learn, joblib, pandas, numpy, matplotlib, seaborn

**RAG (Retrieval Augmented Generation)**
LangChain, Pinecone, HuggingFace embeddings, Groq, RAGAS

**MCP (Model Context Protocol)**
Official MCP Python SDK, Gemini (google-genai), Tavily (web-search)

**User Interface**
Streamlit, streamlit-float

**Authentication & Database**
Supabase, bcrypt


## 8. Running the Project

**1. Clone the project and go into the project folder.**

**2. Create virtual enviroment using uv or any other.**

**3. Install the dependencies:**
```
uv pip install -r requirements.txt
```

**4. Create a `.env` file** in the project root with the following keys filled in:
```
SUPABASE_URL=
SUPABASE_API_KEY=
GEMINI_API_KEY=
TAVILY_API_KEY=
GROQ_API_KEY=
PINECONE_API_KEY=
```

**5. Run the app:**
```
uv run streamlit run app.py
```
The app will open at `http://localhost:8501`

