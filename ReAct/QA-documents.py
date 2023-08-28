from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from pydantic import BaseModel
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import time
import os

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_KEY"] = "<Your api key>"
os.environ["OPENAI_API_BASE"] = "https://<your-resource>.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-05-15" #This might change



class DocSearchInput(BaseModel):
    query: str

class DocSearch:
    def __init__(self, document_texts: List[str]):
        
        
        # Split text based number of tokens (Here chunk size is the number of tokens to split)
        self.text_splitter = TokenTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 10,
            chunk_overlap=0
        )
        self.document_texts = self.text_splitter.create_documents(document_texts)
        

        self.embeddings = OpenAIEmbeddings(
        deployment="<your-deployment-name>",
        model="<your-model-name>", #This can be like "text-embedding-ada-002"
        chunk_size=1 #Because as of now Azure embeddings can embed 1 text at a time
        )
        
        # Because we can embed only one document at a time in Azure, we are iterating over documents, converting it to vectors and storing it one by one
        self.vectorstore = Chroma(embedding_function=self.embeddings)
        for i in range(len(self.document_texts)):
            doc = self.document_texts[i]
            self.vectorstore.add_documents([doc])
            time.sleep(2)

        
        # This is a chat model in Azure
        self.llm = AzureChatOpenAI(
            deployment_name="<your-deployment-name>",
            model_name="<your-model-name>" #The model name can be like gpt-35-turbo-16k
            )
        

    def search(self, query):

       
        ### Retrieval top 5 documents which match the query
        qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}))
        answer = qa_chain({"query": query})
        return answer


class GeneratorOnDocs:

    def __call__(self, user_prompt: str , system_prompt: str, document: str):
        
        docsearch = DocSearch([document])

        tools=[
                Tool.from_function(
                    func=docsearch.search,
                    name="Search in Documentation",
                    description="Useful to lookup technical and non technical documentation",
                    args_schema=DocSearchInput
                    
                )
            ]

        agent = initialize_agent(tools, 
            llm = AzureChatOpenAI(
            deployment_name="<your-deployment-name>",
            model_name="<your-model-name>" #The model name can be like gpt-35-turbo-16k
            ), 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, agent_kwargs ={
                'prefix': system_prompt
            })
        
        for i in agent.run(user_prompt):
            yield i
            
            
gen = GeneratorOnDocs()
document = """Some text which is read from pdf or any other document file"""
user_prompt = """Your user prompt"""
system_prompt = """Your system prompt"""
print(gen(user_prompt, system_prompt, document))
        


