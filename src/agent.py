import os
from langchain.chat_models.openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationSummaryBufferMemory
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.chains import RetrievalQAWithSourcesChain



class SearchAgent:
    def __init__(self):
        
        self.chat_model = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, streaming=True, openai_api_key=os.environ.get("OPENAI_API_KEY"))

        self.vector_store = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai")
        
        self.conversation_memory = ConversationSummaryBufferMemory(llm=self.chat_model, input_key='question', output_key='answer', return_messages=True)
        
        self.google_search = GoogleSearchAPIWrapper()
        
        self.web_research_retriever = WebResearchRetriever.from_llm(vectorstore=self.vector_store, llm=self.chat_model, search=self.google_search)
        
        self.qa_chain = RetrievalQAWithSourcesChain.from_chain_type(self.chat_model, retriever=self.web_research_retriever)
    
    def search_query(self, user_input_question):
        result = self.qa_chain({"question": user_input_question})
        return result["answer"], result["sources"]

