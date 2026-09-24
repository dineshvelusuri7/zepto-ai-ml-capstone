import os
from typing import TypedDict

import chromadb
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")


# --------------------------------------------------
# Models
# --------------------------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_collection(
    name="zepto_policies"
)


# --------------------------------------------------
# State
# --------------------------------------------------

class AssistantState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_chunks: list[str]
    sources: list[str]
    answer: str
    confidence: float


# --------------------------------------------------
# Final response schema
# --------------------------------------------------

class SupportResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float


# --------------------------------------------------
# Node 1: Classify intent
# --------------------------------------------------

def classify_intent(state: AssistantState) -> AssistantState:

    query = state["query"].lower()

    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "track",
        "cancel",
        "cancellation",
        "gift card",
        "support",
        "customer support",
        "delivery fee",
    ]

    if any(keyword in query for keyword in policy_keywords):
        intent = "policy"
    else:
        intent = "general"

    return {
        **state,
        "intent": intent,
    }


# --------------------------------------------------
# Node 2: Retrieve policy and answer
# --------------------------------------------------

def retrieve_and_answer(state: AssistantState) -> AssistantState:

    query = state["query"]

    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    sources = [
        metadata["source"]
        for metadata in metadatas
    ]

    if MOCK_LLM:
        snippet = documents[0][:500]

        answer = (
           f"Based on the retrieved context: {snippet}"
        )
    else:
        context = "\n\n".join(documents)

        prompt = build_prompt(
        query=query,
        context=context,
    )

        answer = call_real_llm(prompt)

    return {
        **state,
        "retrieved_chunks": documents,
        "sources": sources,
        "answer": answer,
        "confidence": 1.0,
    }


# --------------------------------------------------
# Node 3: Direct answer
# --------------------------------------------------

def direct_answer(state: AssistantState) -> AssistantState:

    if MOCK_LLM:
        answer = (
            "I can only answer questions about Zepto "
            "policies right now."
        )
    else:
        answer = (
            "Real LLM generation can be connected here."
        )

    return {
        **state,
        "answer": answer,
        "sources": [],
        "confidence": 1.0,
    }


# --------------------------------------------------
# Conditional routing
# --------------------------------------------------

def route_intent(state: AssistantState) -> str:

    if state["intent"] == "policy":
        return "retrieve_and_answer"

    return "direct_answer"


# --------------------------------------------------
# Build LangGraph
# --------------------------------------------------

builder = StateGraph(AssistantState)

builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_and_answer", retrieve_and_answer)
builder.add_node("direct_answer", direct_answer)

builder.add_edge(START, "classify_intent")

builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)

builder.add_edge("retrieve_and_answer", END)
builder.add_edge("direct_answer", END)

graph = builder.compile()


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def ask_question(query: str) -> SupportResponse:

    result = graph.invoke({
        "query": query
    })

    return SupportResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0),
    )


# --------------------------------------------------
# Local test
# --------------------------------------------------

if __name__ == "__main__":

    response = ask_question(
        "How can I track my order?"
    )

    print("\nANSWER:")
    print(response.answer)

    print("\nSOURCES:")
    print(response.sources)

    print("\nCONFIDENCE:")
    print(response.confidence)



    # --------------------------------------------------
# Structured prompt
# --------------------------------------------------

def build_prompt(query: str, context: str) -> str:

    return f"""
ROLE:
You are a Zepto customer support assistant.

CONTEXT:
Answer using only the retrieved Zepto policy context provided below.

TASK:
Answer the customer's question accurately and clearly.

FORMAT:
Return a short customer-friendly answer.

LENGTH:
Keep the response within 2-4 sentences.

NEGATIVE CONSTRAINT:
Do not invent policies, prices, timelines, services, or guarantees
that are not present in the retrieved context.

FEW-SHOT EXAMPLE:
Customer: How can I track my order?
Context: Every Zepto order shows a live rider-tracking map.
Answer: You can track your order using the live rider-tracking map
available from the Track Order screen.

CUSTOMER QUESTION:
{query}

RETRIEVED POLICY CONTEXT:
{context}
""".strip()


# --------------------------------------------------
# Optional real LLM generation
# --------------------------------------------------

def call_real_llm(prompt: str) -> str:

    import requests

    api_url = os.getenv("LLM_API_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL", "default")

    if not api_url or not api_key:
        raise RuntimeError(
            "Set LLM_API_URL and LLM_API_KEY to use MOCK_LLM=0."
        )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = None

    # Initial attempt + 2 retries
    for attempt in range(3):
        try:
            response = requests.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=30,
            )

            response.raise_for_status()

            data = response.json()

            return data["choices"][0]["message"]["content"]

        except Exception as error:
            last_error = error

            if attempt < 2:
                continue

    raise RuntimeError(
        f"LLM request failed after 2 retries: {last_error}"
    )