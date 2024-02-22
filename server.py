from concurrent import futures
import grpc
import file_processor_pb2
import file_processor_pb2_grpc
from pdfminer.high_level import extract_text
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import numpy as np
import os
import faiss
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Load the tokenizer and model for embeddings
tokenizer_embedding = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
model_embedding = AutoModel.from_pretrained('intfloat/e5-large-v2')

# Load the summarization model and tokenizer
model_name = "facebook/bart-large-cnn"
model_summarizer = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer_summarizer = AutoTokenizer.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model_summarizer, tokenizer=tokenizer_summarizer)


# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        text = extract_text(file_path)
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {e}")

# Function to preprocess text by tokenizing, removing stopwords
def preprocess_text(text):
    if not text:
        raise ValueError("Empty text provided for preprocessing")

    try:
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        processed_tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words and word.isalpha()]
        return ' '.join(processed_tokens)
    except Exception as e:
        raise Exception(f"Error in text preprocessing: {e}")


# Function to summarize text using a pretrained model
def summarize_text(text, max_length=130, min_length=30):
    if not text:
        raise ValueError("Empty text provided for summarization")

    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        raise Exception(f"Error in text summarization: {e}")
    
# Function to average pool the embeddings for attention masking
def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# Function to vectorize text using embeddings
def vectorize_text(text: str) -> torch.Tensor:
    inputs = tokenizer_embedding(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_embedding(**inputs)
    embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
    return F.normalize(embeddings, p=2, dim=1).squeeze()

# Function to build an index for vectors using Faiss
def build_index(vectors):
    if not vectors:
        raise ValueError("Empty vector list provided for index building")

    try:
        dimension = len(vectors[0])
        index = faiss.IndexFlatL2(dimension)
        vectors_np = np.array(vectors).astype('float32')
        index.add(vectors_np)
        return index
    except Exception as e:
        raise Exception(f"Error building index with Faiss: {e}")

# Function to process a file by extracting, preprocessing, summarizing, vectorizing text, and building an index
def process_file(file_path):
    try:
        text = extract_text_from_pdf(file_path)
        preprocessed_text = preprocess_text(text)
        summarized_text = summarize_text(preprocessed_text)
        vectors = vectorize_text(summarized_text)
        index = build_index(vectors)
        return summarized_text, index
    except Exception as e:
        return str(e), None

# gRPC server implementation with methods for processing PDFs and summarizing text
class FileProcessorServicer(file_processor_pb2_grpc.FileProcessorServicer):
    def ProcessPdf(self, request, context):
        try:
            file_content = request.content
            text = extract_text_from_pdf(file_content)
            preprocessed_text = preprocess_text(text)
            summary = summarize_text(preprocessed_text)
            return file_processor_pb2.SummaryResponse(summary=summary)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details('Error processing PDF: ' + str(e))
            return file_processor_pb2.SummaryResponse()

    def SearchDocuments(self, request, context):
        try:
            query = request.query
            query_vector = vectorize_text(query)
            documents = [] 
            return file_processor_pb2.SearchResponse(documents=documents)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details('Error searching documents: ' + str(e))
            return file_processor_pb2.SearchResponse()
        
    def SummarizeText(self, request, context):
        try:
            text = request.text
            if not text.strip():
                raise ValueError("Empty text provided for summarization")

            summary = summarize_text(text)
            return file_processor_pb2.SummaryResponse(summary=summary)
        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return file_processor_pb2.SummaryResponse()
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details('Error in text summarization: ' + str(e))
            return file_processor_pb2.SummaryResponse()
     
        
# Function to start and run the gRPC server
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    file_processor_pb2_grpc.add_FileProcessorServicer_to_server(
        FileProcessorServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()