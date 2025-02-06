import pandas as pd
import json
import openai
from dotenv import load_dotenv
import os

# Carregar a API Key do arquivo .env
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# 1️⃣ Carregar o arquivo CSV
arquivo_csv = 'tuning.csv'  # Substitua pelo nome do seu arquivo
df = pd.read_csv(arquivo_csv)

# 2️⃣ Converter para JSONL
arquivo_jsonl = 'dados_finetuning.jsonl'

with open(arquivo_jsonl, 'w', encoding='utf-8') as f:
    for index, row in df.iterrows():
        data = {
            "messages": [
                {"role": "user", "content": row['Perguntas']},
                {"role": "assistant", "content": row['Respostas']}
            ]
        }
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"Arquivo {arquivo_jsonl} criado com sucesso!")

# 3️⃣ Upload do arquivo para o OpenAI
upload_response = openai.File.create(
    file=open(arquivo_jsonl, "rb"),
    purpose='fine-tune'
)
file_id = upload_response['id']

print(f"Arquivo enviado com sucesso! File ID: {file_id}")

# 4️⃣ Iniciar o Fine-Tuning
fine_tune_response = openai.FineTuningJob.create(
    training_file=file_id,
    model="gpt-3.5-turbo"
)

print(f"Fine-tuning iniciado! Job ID: {fine_tune_response['id']}")
