from key import api_key
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import getpass
import os


os.environ["GOOGLE_API_KEY"] = api_key

llm = ChatGoogleGenerativeAI(api_key=os.getenv('GOOGLE_API_KEY'), model="gemini-1.5-pro-latest")
embeddings = HuggingFaceEmbeddings()


vectordb_file_path = "faiss_vectordb"

def create_vectordb():

    loader = CSVLoader(file_path = 'train.csv', source_column = "qtype",  encoding='utf-8')
    data = loader.load()
    vectordb = FAISS.from_documents(documents = data, embedding = embeddings)
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization = True)

    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate a direct answer based on this context only.
    If the answer is not found in the context, kindly state "I don't know." Do not provide any additional explanation or context beyond the direct answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return qa_chain


if __name__ == "__main__":

    # create_vectordb()
    qa_chain = get_qa_chain()

    print(qa_chain("what are marine toxins?"))