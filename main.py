from src.agent import SearchAgent
from dotenv import load_dotenv

load_dotenv()

agent = SearchAgent()
user_input_question = input("Ask a question: ")

answer, sources = agent.search_query(user_input_question)
print("Answer:", answer)
print("Sources:", sources)