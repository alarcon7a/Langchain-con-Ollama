from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from upload_data import cargar_documentos, crear_vectorstore
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Códigos de escape ANSI para colores
AZUL = "\033[94m"
VERDE = "\033[92m"
RESET = "\033[0m"
def iniciar_chat(ruta_archivo):
    llm = Ollama(model="llama3")
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    vectorstore = Chroma(embedding_function=embed_model,
                                       persist_directory="chroma_db_dir",
                                       collection_name="stanford_report_data")
    total_rows = len(vectorstore.get()['ids'])
    if total_rows == 0:
        docs = cargar_documentos(ruta_archivo)
        vectorstore = crear_vectorstore(docs)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

    custom_prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else but in spanish.
    Helpful answer:
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    print("¡Bienvenido al chat! Escribe 'salir' para terminar.")
    while True:
        pregunta = input(f"{AZUL}Tú:{RESET} ")
        if pregunta.lower() == 'salir':
            print("¡Hasta luego!")
            break

        respuesta = qa.invoke({"query": pregunta})
        metadata = []
        for _ in respuesta['source_documents']:
            metadata.append(('page: '+str(_.metadata['page']), _.metadata['file_path']))
        print(f"{VERDE}Asistente:{RESET}", respuesta['result'], '\n', metadata)

if __name__ == "__main__":
    ruta_archivo = "src/HAI_2024_AI-Index-Report.pdf"
    iniciar_chat(ruta_archivo)