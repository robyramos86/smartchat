from glob import glob
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os
import zipfile



os.environ["OPENAI_API_KEY"] = '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'



def construct_index():
    # extrair arquivos do zip
    zip_path = "tema.zip"
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
    except Exception as e:
        print("Erro ao extrair o arquivo zip:", e)

        
    max_input_size = 3500
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(".").load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index


def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')

    # Ler e concatenar os documentos da pasta "docs" como o contexto relevante
    documents = ""
    for file_path in glob(os.path.join(".", "*.{txt,pdf}")):
        with open(file_path, "r") as f:
            documents += f.read() + " "
    contexto = documents.strip()

    # Combinar o contexto e a pergunta de entrada
    with open('tema.txt', 'r') as f:
        texto_prefixo = f.readline().strip()
    texto_entrada = f"Dentro do assunto {texto_prefixo} me responda: {input_text}{contexto} se não for {texto_prefixo} não responda"
    print(texto_entrada)
      
    response = index.query(texto_entrada, response_mode="compact")
    return response.response


description = """
A IA foi treinada com materiais enviados e responde perguntas sobre o tema definido!
"""

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Como podemos te ajudar?"),
                     outputs="text",
                     description=description,                     
                     title="Demonstração Chat OpenAI")


index = construct_index()
iface.launch(share=False)