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

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <pdf_path>")
        return
    
    pdf_path = sys.argv[1]
    if not os.path.isfile(pdf_path):
        print(f"Error: '{pdf_path}' is not a valid file.")
        return

    # --- Step 1: Document Processing (Run once) ---
    stop_animation = False
    def animate(message="Loading document"):
        for c in itertools.cycle(['|', '/', '-', '\\']):
            if stop_animation:
                break
            sys.stdout.write(f'\r{message} {c}')
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\r' + ' ' * (len(message) + 2) + '\r')

    t = threading.Thread(target=animate)
    t.start()

    try:
        loader = PyPDFLoader(file_path=pdf_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            collection_name="local-rag"
        )
        
        llm = ChatOllama(model="llama3", temperature=0)

        # RAG Chain Setup
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

        history_aware_retriever = create_history_aware_retriever(
            llm, vector_db.as_retriever(), contextualize_q_prompt
        )

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

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    finally:
        stop_animation = True
        t.join()

    # --- Step 2: Interactive Chat Loop ---
    chat_history = []
    print(f"\n✅ Ready! Chatting with: {os.path.basename(pdf_path)}")
    print("(Type 'exit' or 'quit' to stop)\n")

    while True:
        query = input("You: ").strip()
        
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        if not query:
            continue

        # Start loading animation for the LLM response
        stop_animation = False
        t = threading.Thread(target=animate, args=("Thinking",))
        t.start()

        try:
            response = rag_chain.invoke({
                "input": query, 
                "chat_history": chat_history
            })
            
            answer = response["answer"]
            
            # Update history to maintain context
            chat_history.append(("human", query))
            chat_history.append(("assistant", answer))

        finally:
            stop_animation = True
            t.join()

        print(f"AI: {answer}\n")

if __name__ == "__main__":
    main()