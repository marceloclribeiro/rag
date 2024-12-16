# README.md

## Instalação de Dependências
Para instalar as dependências do projeto, siga os passos abaixo:

1. Certifique-se de ter o Python 3.8 ou superior instalado em sua máquina.
2. Crie e ative um ambiente virtual (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Para Linux/MacOS
   venv\Scripts\activate    # Para Windows
   ```
3. Instale as dependências listadas no arquivo `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Como Rodar o Código

1. Certifique-se de que as dependências foram instaladas corretamente.
2. Coloque seus arquivos PDF na pasta `data`.
3. Execute o script principal do projeto:
   ```bash
   streamlit run app.py
   ```

4. A aplicação estará disponível em seu navegador no endereço `http://localhost:8501`.
5. Insira uma pergunta ou processe PDFs diretamente pela interface da aplicação.

### Configuração da API do GenAI
Certifique-se de configurar a chave de API no arquivo .env
GENAI_API_KEY = SUA_CHAVE_DE_API
Substitua `'SUA_CHAVE_DE_API'` pela sua chave de API válida.