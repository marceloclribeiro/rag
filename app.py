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

# Configura a chave de API do modelo de geração
api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    raise ValueError("A chave de API (GENAI_API_KEY) não foi encontrada no arquivo .env")
genai.configure(api_key=api_key)

def extract_text_from_pdf(pdf_path):
    # Extrai o texto de todas as páginas de um arquivo PDF, removendo quebras de linha
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text.replace("\n", "")

def query_chromadb(collection, embedding):
    # Realiza consulta por similaridade no ChromaDB, retornando documentos, metadados e distâncias
    results = collection.query(embedding, n_results=15)
    return results["documents"], results["metadatas"], results["distances"]

def embedding_gen_ai(text):
    # Gera embedding do texto usando o modelo GenAI
    result = genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_query")
    return result['embedding']

def format_prompt(prompt_template, query, chunks):
    # Formata o prompt substituindo placeholders pela consulta do usuário e conteúdo relevante
    return prompt_template.format(query=query, chunks=chunks)

def split_text_into_chunks(text, max_length=1000, sentence_max_length=2000):
    # Divide o texto em chunks menores para processamento de embeddings, respeitando tamanho máximo
    sentence_pattern = re.compile(r'([^.!?]+[.!?])')
    sentences = sentence_pattern.findall(text)

    chunks = []
    current_chunk = []
    current_length = 0

    def add_chunk(chunk_list):
        if chunk_list:
            chunks.append("".join(chunk_list).strip())

    for sentence in sentences:
        # Caso a sentença seja muito longa, é dividida em partes menores
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
    # Processa todos os PDFs em uma pasta, gerando embeddings e salvando-os na coleção do ChromaDB
    all_chunks = {}
    processed_any = False
    processed_pdfs = []

    for pdf_file in os.listdir(folder_path):
        if pdf_file.endswith(".pdf"):
            pdf_name = os.path.splitext(pdf_file)[0]
            pdf_path = os.path.join(folder_path, pdf_file)

            # Verifica se o PDF já foi processado previamente
            existing_docs = collection.get(ids=[f"{pdf_name}_doc_0"])
            if existing_docs["ids"]:
                continue

            # Extrai texto e cria chunks
            text = extract_text_from_pdf(pdf_path)
            chunks = split_text_into_chunks(text)

            # Gera e adiciona embeddings à coleção
            for i, chunk in enumerate(chunks):
                doc_id = f"{pdf_name}_doc_{i}"
                embedding = embedding_gen_ai(chunk)
                collection.add(
                    documents=[chunk],
                    metadatas=[{"chunk_id": i, "pdf_name": pdf_name}],
                    ids=[doc_id],
                    embeddings=[embedding]
                )

            all_chunks[pdf_name] = chunks
            processed_pdfs.append(pdf_name)
            processed_any = True

    # Salva chunks em JSON para referência
    with open("chunks.json", "w") as file:
        json.dump(all_chunks, file)

    # Retorna status do processamento
    if processed_any:
        return f"Processados os seguintes PDFs: {', '.join(processed_pdfs)}"
    else:
        return "Todos os PDFs já foram processados."

def process_query(collection, query, model_gen):
    # Processa a consulta do usuário, recupera documentos similares, gera prompt e obtém resposta do modelo
    with open("prompt_template.yml", "r") as file:
        prompts = yaml.safe_load(file)

    system_prompt = prompts["System_Prompt"]
    prompt_template = prompts["prompt_instructions"]

    embedding = embedding_gen_ai(query)
    documents, metadatas, distances = query_chromadb(collection, embedding)

    # Trata documentos para formar o contexto
    flattened_documents = []
    for doc in documents:
        if isinstance(doc, list):
            flattened_documents.append(" ".join(map(str, doc)))
        else:
            flattened_documents.append(str(doc))

    chunks_text = "\n\n".join(flattened_documents)
    prompt = format_prompt(prompt_template, query, chunks_text)

    response = model_gen.generate_content(prompt)
    return response.text

def main():
    # Função principal que inicializa interface em Streamlit, gerencia entrada do usuário e exibe respostas
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