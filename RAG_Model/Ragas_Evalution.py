from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from ragas.run_config import RunConfig


load_dotenv()

judge_llm = LangchainLLMWrapper(
     ChatGroq(
        model="openai/gpt-oss-120b",
        temperature = 0.5
    )
)

judge_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
)

eval_data = {
    "question" : ["What is a Systematic Investment Plan (SIP)?",

                  "What is the difference between simple interest and compound interest?",

                  "What is the Debt-to-Income (DTI) ratio used for in loan approval?",

                  "What is an Equated Monthly Installment (EMI)?",

                  "What is diversification in investing?",
                  ],

    "answer" : [
        """
        A SIP is a method of investing a fixed amount in a mutual fund 
        at regular intervals, such as monthly, instead of investing 
        a lump sum at once.
        """,

        """
        Simple interest is calculated only on the original principal, 
        while compound interest is calculated on the principal plus 
        previously accumulated interest.
        """,

        """
        The DTI ratio is used to measure a borrower's ability to manage 
        monthly debt payments relative to their income, helping lenders 
        assess loan repayment capacity.
        """,

        """
        An EMI is a fixed monthly payment made by a borrower to a lender 
        on a specified date each month, covering both principal and 
        interest, until the loan is fully repaid.
        """,

        """
        Diversification means spreading investments across different asset classes or
        securities to reduce overall risk.
        """
    ],

    "contexts" : [

        ["""
        A Systematic Investment Plan allows an investor to invest a fixed sum regularly 
        (weekly, monthly, or quarterly) into a chosen mutual fund scheme. It helps average
        out the purchase cost over time through rupee-cost averaging and encourages disciplined, long-term investing.
        """
        ],

        ["""
        Simple interest is computed as Principal x Rate x Time, applied only to the original amount borrowed or invested.
        Compound interest, on the other hand, is calculated on the initial principal and also on the accumulated interest 
        from previous periods, causing the amount to grow faster over time.
        """
        ],

        ["""
        The Debt-to-Income ratio is calculated by dividing a borrower's total monthly debt payments by their gross monthly income.
        Lenders use this ratio to evaluate whether an applicant can comfortably take on additional debt. A lower DTI ratio generally 
        indicates a good balance between debt and income, improving the chances of loan approval.
        """
        ], 

        ["""
        An EMI, or Equated Monthly Installment, is the fixed amount a borrower pays every month toward a loan. 
        Each EMI payment consists of two components: a portion that goes toward the loan principal and a portion that goes toward interest.
        Over the loan tenure, the interest component decreases while the principal component increases.
        """
        ],

        ["""
        Diversification is a risk management strategy that mixes a variety of investments, such as stocks, bonds, and real estate, within a portfolio. 
        The rationale is that a portfolio built with different asset types will, on average, yield higher long-term returns and lower the risk of any single investment's poor performance.
        """
        ], 

],

    "ground_truth" : [

        """
        A SIP is a way of investing a fixed sum of money in a mutual fund at regular intervals, helping investors average their purchase cost over time through disciplined, periodic investing.
        """,

        """
        Simple interest applies only to the original principal amount, whereas compound interest applies to both the principal and the interest already earned, resulting in faster growth over time.
        """,

        """
        The DTI ratio measures the proportion of a borrower's monthly income that goes toward debt payments, and lenders use it to assess a borrower's capacity to take on and repay additional debt.
        """, 

        """
        An EMI is a fixed monthly payment consisting of both principal and interest components that a borrower pays to gradually repay a loan over its tenure.
        """,

        """
        Diversification is an investment strategy of spreading capital across different asset classes or securities to reduce the risk of loss from any single investment.
        """,
        ]
}

dataset = Dataset.from_dict(eval_data)

run_config = RunConfig(
    timeout=120,
    max_retries=3,
    max_workers=1
)


result = evaluate(
    dataset=dataset,

    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],

    llm=judge_llm,
    embeddings=judge_embeddings,
    run_config=run_config
)


df = result.to_pandas()
print(df)

df.to_csv(
    "ragas_financial_result.csv",
    index=False
)

print("\nResults saved to: ragas_financial_result.csv")

