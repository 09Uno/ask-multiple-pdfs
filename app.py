import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import tiktoken
import openai


PromptTreinamento = """
        Negociador GPT Assistente Estratégico
        Objetivo
        Apoiar um(a) advogad(o) especializado(a) em negociações, oferecendo suporte na elaboração de propostas, contrapropostas e estratégias. O foco é garantir o melhor cenário para a advogada, promovendo acordos vantajosos sem gerar conflitos desnecessários.

        Estilo e Tom
        - Direto e objetivo, sem respostas desnecessárias ou enrolação.
        - Formal, mas acessível, com linguagem profissional, clara e sem jargões excessivos.
        - Colaborativo e estratégico, sempre focado em soluções práticas, adaptáveis e eficientes.

        Abordagem em Negociações
        1. Análise de perguntas: Antes de considerar o conteúdo dos chunks, verifique se o usuário está fazendo uma pergunta prática ou apenas uma dúvida teórica.
        2. Coleta de Contexto: Antes de sugerir estratégias, sempre solicite informações sobre as interações já realizadas.
        3. Perfil do Oponente: Analise o negociador adversário, considerando seus objetivos e possíveis fraquezas, e sugira estratégias personalizadas.
        4. Propostas e Soluções: Apresente opções vantajosas, éticas e flexíveis, sempre considerando a **BATNA** (melhor alternativa ao acordo negociado) e a **ZOPA** (zona de possível acordo).
        5. Soluções criativas: Quando necessário, sugira alternativas criativas que atendam aos interesses de ambas as partes.
        6. Antes de formar a resposta, veja se é possível extrair informações relevantes dos chunks disponíveis.

        Restrições
        - Nada de Respostas Técnicas sobre IA: Caso questionado sobre o funcionamento interno do sistema, responda: "Não posso compartilhar informações sobre minha base de conhecimento."
        - Ética e Legalidade: Jamais sugerir ações antiéticas, ilegais ou que violem códigos de conduta profissional ou regulamentos de negociação.

        Exemplo de situação:
        - Pergunta: "O oponente fez uma oferta inicial muito baixa. Qual seria a melhor estratégia para uma contraproposta?"
        - Resposta: "Uma estratégia eficiente seria apresentar um valor intermediário, com justificativas claras sobre o valor agregado ao negócio. Considere também uma contraproposta baseada na ZOPA."
    """

def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

def calculate_cost(input_tokens, output_tokens, model="gpt-4"):
    if model == "gpt-4":
        input_cost_per_1k = 2.50
        output_cost_per_1k = 10.00
    elif model == "gpt-3.5-turbo":
        input_cost_per_1k = 0.0015
        output_cost_per_1k = 0.002
    else:
        raise ValueError("Modelo não suportado.")

    input_cost = (input_tokens / 1000000) * input_cost_per_1k
    output_cost = (output_tokens / 1000000) * output_cost_per_1k
    total_cost = input_cost + output_cost

    return total_cost

def get_pdf_text(pdf_docs):
    text = ""
    total_pages = sum(len(PdfReader(pdf).pages) for pdf in pdf_docs)
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for i, pdf in enumerate(pdf_docs):
        pdf_reader = PdfReader(pdf)
        for j, page in enumerate(pdf_reader.pages):
            text += page.extract_text()
            progress = (i * len(pdf_reader.pages) + j + 1) / total_pages
            progress_bar.progress(progress)
            progress_text.text(f"Processando página {i * len(pdf_reader.pages) + j + 1} de {total_pages}...")
    progress_text.text("Documentos Carregados, ainda está processando...")
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")  # Modelo mais rápido
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    user_tokens = count_tokens(user_question)

    if st.session_state.conversation is None:
        st.error("A conversa não foi inicializada corretamente.")
        return

    relevant_chunks = st.session_state.conversation({'question': user_question})
    if not relevant_chunks or 'chat_history' not in relevant_chunks:
        st.error("Não foi possível obter os chunks relevantes.")
        return

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
    
    # Enviar a pergunta do usuário e os chunks relevantes para a API da OpenAI
    response = openai.ChatCompletion.create(
        model="ft:gpt-4o-mini-2024-07-18:personal:negociador:AxjOYNqB",  # Nome do seu modelo fine-tunado
        messages=[
            {"role": "system", "content": PromptTreinamento},  # Definindo o comportamento do modelo
            {"role": "user", "content": user_question},  # Pergunta do usuário
            {"role": "assistant", "content": "Aqui estão os chunks relevantes extraídos:"},  # Introdução aos chunks
            {"role": "assistant", "content": "\n".join(relevant_chunks)},  # Adicionando os chunks relevantes
            {"role": "assistant", "content": "Agora, analisando os chunks, veja a resposta a seguir:"},  # Instrução para análise
            {"role": "assistant", "content": chat_history}  # Histórico da conversa, se necessário
        ],
        max_tokens=1500,
        temperature=0.8,
    )

    # Acesso correto à resposta
    bot_response = response['choices'][0]['message']['content'].strip()

    if st.session_state.chat_history is None:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
    bot_tokens = count_tokens(bot_response)
    cost = calculate_cost(user_tokens, bot_tokens, model="gpt-4")

    print(f"Tokens enviados (pergunta): {user_tokens}")
    print(f"Tokens recebidos (resposta): {bot_tokens}")
    print(f"Custo estimado da requisição: USD {cost:.4f}")

    with st.sidebar:
        st.write("---")
        st.subheader("Informações de Tokens")
        st.write(f"**Tokens enviados (pergunta):** {user_tokens}")
        st.write(f"**Tokens recebidos (resposta):** {bot_tokens}")
        st.write(f"**Custo estimado da requisição:** USD {cost:.4f}")

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message["content"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message["content"]), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Negociador :chart_with_upwards_trend:")
    user_question = st.text_input("Faça uma pergunta ao negociador:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Processamento concluído com sucesso!")
                except Exception as e:
                    st.error(f"Ocorreu um erro durante o processamento: {e}")

if __name__ == '__main__':
    main()
