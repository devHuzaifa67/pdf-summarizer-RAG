from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
  persist_directory = "chroma_db",
  embedding_function = embedding_model
)

retriever = vectorstore.as_retriever(
  search_type = "mmr",
  search_kwargs = {
    "k": 5, 
    "fetch_k": 20,
    "lambda_mult" : 0.1
  }
)

llm = ChatMistralAI(model="mistral-small-2506")

# Create a prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """YYou are a helpful assistant. Use the provided context if relevant, otherwise answer from general knowledge.
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("Rag system created")

print("press 0 to exit")

while True:
  query =input("You: ")
  if query == "0":
    break

  docs = retriever.invoke(query)

  context ="\n\n".join(
    [doc.page_content for doc in docs]
  )

  final_prompt = prompt.invoke({
    "context": context,
    "question": query
  }
)

  response = llm.invoke(final_prompt)

  print(f"\n AI: {response.content}")
