import os
import sys
import itertools
import threading
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama



def main():
    
    if len(sys.argv) > 2:
        pdf_path = sys.argv[1]
        query = sys.argv[2]
        if not os.path.isfile(pdf_path):
            print(f"'{pdf_path}' isn't a valid pdf_path.")
            return
    else:
        print("Usage: python mainn.py <pdf_path> <query>")
        return

    done = False
    #here is the animation
    def animate():
        for c in itertools.cycle(['|', '/', '-', '\\']):
            if done:
                break
            sys.stdout.write('\rloading ' + c)
            sys.stdout.flush()
            time.sleep(0.1)

    t = threading.Thread(target=animate)
    t.start()


    # Load the pdf
    loader = PyPDFLoader(file_path=pdf_path)
    data = loader.load()

    # split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    #
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        collection_name="local-rag"
    )
    
    # Initialize our Local LLM
    llm = ChatOllama(model="llama3", temperature=0)

    # Reasoning Step: Contextualize the Question
    # This sub-chain re-writes the user question to be standalone if there's chat history.
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Create the history-aware retriever
    # This uses the vector_db from Step 3
    history_aware_retriever = create_history_aware_retriever(
        llm, vector_db.as_retriever(), contextualize_q_prompt
    )

    # Answer Generation Step
    # This defines how the final answer is crafted using the retrieved chunks.
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the user's question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # This chain "stuffs" the retrieved documents into the prompt
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # The Final Retrieval Chain
    # This unites the history-aware retriever and the document chain.
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    # query = "Which college did he study and which branch ?"
    chat_history = []
    response = rag_chain.invoke({
        "input": query, 
        "chat_history": chat_history
    })
    
    # Update history for the next turn
    chat_history.extend([
        ("human", query),
        ("assistant", response["answer"])
    ])

    answer, sources = response["answer"], response["context"]
    done = True

    sys.stdout.write('\r')
    print(f">>>> {answer}")

if __name__ == "__main__":
    main()
