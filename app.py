import yaml
from pypdf import PdfReader
import chromadb
import json
import re
import os
import streamlit as st
from dotenv import load_dotenv

import google.generativeai as genai
load_dotenv()

api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    raise ValueError("A chave de API (GENAI_API_KEY) não foi encontrada no arquivo .env")
genai.configure(api_key=api_key)

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text.replace("\n", "")

def query_chromadb(collection, embedding):
    results = collection.query(embedding, n_results=15)
    return results["documents"], results["metadatas"], results["distances"]

def embedding_gen_ai(text):
    result = genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_query")
    return result['embedding']

def format_prompt(prompt_template, query, chunks):
    return prompt_template.format(query=query, chunks=chunks)

def split_text_into_chunks(text, max_length=1000, sentence_max_length=2000):
    sentence_pattern = re.compile(r'([^.!?]+[.!?])')
    sentences = sentence_pattern.findall(text)

    chunks = []
    current_chunk = []
    current_length = 0

    def add_chunk(chunk_list):
        if chunk_list:
            chunks.append("".join(chunk_list).strip())

    for sentence in sentences:
        if sentence_max_length and len(sentence) > sentence_max_length:
            parts = [sentence[i:i+sentence_max_length] for i in range(0, len(sentence), sentence_max_length)]
            for part in parts:
                if current_length + len(part) > max_length:
                    add_chunk(current_chunk)
                    current_chunk = []
                    current_length = 0
                current_chunk.append(part)
                current_length += len(part)
        else:
            if current_length + len(sentence) > max_length:
                add_chunk(current_chunk)
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += len(sentence)

    add_chunk(current_chunk)
    return chunks
def process_pdfs_in_folder(folder_path, collection):
    """
    Processa os PDFs na pasta e retorna mensagens de status sobre os processados.

    Args:
        folder_path (str): Caminho da pasta contendo os PDFs.
        collection (object): Coleção para armazenar os embeddings.

    Returns:
        str: Mensagem indicando quais PDFs foram processados ou se todos já foram processados.
    """
    all_chunks = {}  # Dicionário para armazenar os chunks de todos os PDFs
    processed_any = False  # Flag para verificar se algum PDF foi processado
    processed_pdfs = []  # Lista para armazenar nomes dos PDFs processados

    for pdf_file in os.listdir(folder_path):
        if pdf_file.endswith(".pdf"):
            pdf_name = os.path.splitext(pdf_file)[0]  # Nome do PDF sem a extensão
            pdf_path = os.path.join(folder_path, pdf_file)

            # Verifica se o PDF já foi processado
            existing_docs = collection.get(ids=[f"{pdf_name}_doc_0"])
            if existing_docs["ids"]:  # Verifica se o ID do primeiro chunk já existe
                continue

            # Extrai texto e divide em chunks
            text = extract_text_from_pdf(pdf_path)
            chunks = split_text_into_chunks(text)

            for i, chunk in enumerate(chunks):
                doc_id = f"{pdf_name}_doc_{i}"

                # Gera e armazena o embedding
                embedding = embedding_gen_ai(chunk)
                collection.add(
                    documents=[chunk],
                    metadatas=[{"chunk_id": i, "pdf_name": pdf_name}],
                    ids=[doc_id],
                    embeddings=[embedding]
                )

            # Adiciona os chunks ao dicionário
            all_chunks[pdf_name] = chunks
            processed_pdfs.append(pdf_name)  # Adiciona o PDF à lista de processados
            processed_any = True

    # Salva todos os chunks em um arquivo JSON para referência futura
    with open("chunks.json", "w") as file:
        json.dump(all_chunks, file)

    if processed_any:
        return f"Processados os seguintes PDFs: {', '.join(processed_pdfs)}"
    else:
        return "Todos os PDFs já foram processados."

def process_query(collection, query, model_gen):

    with open("prompt_template.yml", "r") as file:
        prompts = yaml.safe_load(file)

    system_prompt = prompts["System_Prompt"]
    prompt_template = prompts["prompt_instructions"]
    # Gera o embedding da consulta
    embedding = embedding_gen_ai(query)
    
    # Consulta ao ChromaDB
    documents, metadatas, distances = query_chromadb(collection, embedding)

    # Flatten dos documentos
    flattened_documents = []
    for doc in documents:
        if isinstance(doc, list):  # Se o documento for uma lista aninhada
            flattened_documents.append(" ".join(map(str, doc)))
        else:  # Documento já é uma string
            flattened_documents.append(str(doc))

    # Junta os documentos em um único texto
    chunks_text = "\n\n".join(flattened_documents)

    # Formata o prompt
    prompt = format_prompt(prompt_template, query, chunks_text)

    # Gera a resposta usando o modelo
    response = model_gen.generate_content(prompt)

    # Retorna o texto gerado
    return response.text

def main():
    model_gen = genai.GenerativeModel('gemini-1.5-flash')
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection("pdf_embeddings")

    st.set_page_config(
        page_title="Pelot.ai - Descubra Pelotas",
        page_icon="🧁"
    )
    st.markdown(
        """
        <style>

        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');

        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif; 
            font-size: 22px;
            font-weight: 700;
            color: #091747;
        }
        .stTextInput > div:first-child {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 5px;
            padding: 5px;
            color: black;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Pelot.ai 👸🧁🐜")
    st.write("Faça perguntas sobre a história da cidade de Pelotas!")

    question = st.text_input(
        "### Insira aqui sua pergunta:", placeholder="Digite sua pergunta aqui..."
    )
    
    if st.button("Enviar"):
        if question.strip():
            response = process_query(collection, question, model_gen)
            st.markdown("### Resposta:")
            st.write(response)
        else:
            st.warning("Faça uma pergunta!.")

    else:
        st.markdown("### Resposta:")
        st.write("Sua resposta aparecerá aqui....")
    
    if st.button("Processar PDFs"):
        with st.spinner("Processando PDFs. Por favor, aguarde..."):
            folder_path = "data" 
            response = process_pdfs_in_folder(folder_path, collection)
        st.markdown("### Status do Processamento de PDFs:")
        st.write(response)


if __name__ == "__main__":
    main()