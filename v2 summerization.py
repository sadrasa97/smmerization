import torch.nn.functional as F
import pandas as pd
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, pipeline
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
from scipy import spatial
import openai
import ast
import tiktoken
import pypdf
# models
GPT_MODEL = "gpt-3.5-turbo"
MODEL_NAME = 'intfloat/e5-large-v2'
openai.api_key = "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"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to summarize text
def summarize_text(text):
    summary_result = summarizer(text, max_length=512, min_length=50, do_sample=False)
    return summary_result[0]['summary_text'] if summary_result else text

# Function to get embeddings
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_embedding(text):
    input_texts = ['passage: '+text]
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**batch_dict)
    return average_pool(outputs.last_hidden_state, batch_dict['attention_mask']).tolist()[0]

def pdf_to_pages(file):
    "extract text (pages) from pdf file"
    pages = []
    pdf = pypdf.PdfReader(file)
    for p in range(len(pdf.pages)):
        page = pdf.pages[p]
        text = page.extract_text()
        text = text.replace('\n', '')
        # کاهش طول جمله به 512 کاراکتر
        text = text[:2000]
        pages += [text]
    return pages

def choose_file():
    Tk().withdraw() 
    return askopenfilename() 

# Get the input file path from the user
file = choose_file()

# Get the file extension
extension = os.path.splitext(file)[1]

if extension == '.pdf':
    text = pdf_to_pages(file)
    embeddings = []
    for batch_start in range(0, len(text),):
        batch = text[batch_start]
        # Summarize the batch
        summary = summarize_text(batch)
        # Get the embedding of the summary
        embeddings.append(get_embedding(summary))

    df = pd.DataFrame({"text":text, "embedding": embeddings})
    df.to_csv('exonai.csv')

elif extension == '.csv':
    df = pd.read_csv(file)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 5
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding = get_embedding(query)
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = '"'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\narticle section:\n"""\n{string}\n"""'
        if (
            tiktoken.Tokenizer().count_tokens(message + next_article + question) > token_budget
             ):
                break
        else:
            message += next_article
        return message + question

def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about the pdf."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message

while True:
    query = input("Please enter your question :\n ")
    print(ask(query))
    if query.lower() == 'exit':
        break
