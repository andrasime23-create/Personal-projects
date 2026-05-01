import os
import requests
import PyPDF2
from tqdm.auto import tqdm
import random
import pandas as pd
from spacy.lang.en import English
from sentence_transformers import util, SentenceTransformer
import torch
import numpy as np
from time import perf_counter as timer
import textwrap3
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.utils import is_flash_attn_2_available 
from transformers import BitsAndBytesConfig



os.environ['HTTP_PROXY'] = 'secret'
os.environ['HTTPS_PROXY'] = 'secret'

# Get PDF document path
pdf_path = "C:\\Users\\SIN8CLJ\\Desktop\\Work\\MFD\\BEFORE_You_Get_Your_Puppy_Dr.pdf"

# Download PDF
if not os.path.exists(pdf_path):
    print("[INFO] File doesn't exist, downloading...")

    # Enter the URL of the PDF
    url = "https://www.dogstardaily.com/files/downloads/BEFORE_You_Get_Your_Puppy.pdf"

    # The local filename to save the downloaded file
    filename = pdf_path

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open the file and save it
        with open(filename, "wb") as file:
            file.write(response.content) 
        print(f"[INFO] The file has been downloaded and saved as {filename}")
    else:
        print(f"[INFO] Failed to download the file. Status code: {response.status_code}")

else:
    print(f"[INFO] File {pdf_path} exists.")


def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()

    # Potentially more text formatting functions can go here
    return cleaned_text

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    pages_and_texts = []
    
    # Open the PDF file
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        
        # Iterate over each page in the PDF
        for page_number, page in tqdm(enumerate(reader.pages)):
            text = page.extract_text()
            text = text_formatter(text=text)
            pages_and_texts.append({
                "page_number": page_number + 1,
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) / 4,  # 1 token = ~4 characters
                "text": text
            })
    
    return pages_and_texts

# Read and process the PDF
pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)

# Output the first two entries
print(pages_and_texts[:2])


#Splitting pages into sentences
nlp = English()

# Add a sentencizer pipeline
nlp.add_pipe("sentencizer")

pages_and_texts[30]
for item in tqdm(pages_and_texts):
    item["sentences"] = list(nlp(item["text"]).sents)

    # Make sure all sentences are strings (the default type is a spaCy datatype)
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]

    # Count the sentences
    item["page_sentence_count_spacy"] = len(item["sentences"])

#Show the functionality of the sampling of sentences
print(random.sample(pages_and_texts, k=1))

# Define split size to turn groups of sentences into chunks
num_sentence_chunk_size = 10
# Create a function to split lists of texts recursively into chunk size
def split_list(input_list: list[str],
               slice_size: int=num_sentence_chunk_size) -> list[list[str]]:
    return [input_list[i:i+slice_size] for i in range(0, len(input_list), slice_size)]

test_list = list(range(25))
split_list(test_list)

# Loop through pages and texts and split sentences into chunks
for item in tqdm(pages_and_texts):
    item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                         slice_size=num_sentence_chunk_size)
    item["num_chunks"] = len(item["sentence_chunks"])


#Splittin each chunk into its own item
pages_and_chunks = []
for item in tqdm(pages_and_texts): 
    for sentence_chunk in item["sentence_chunks"]: 
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]

        # Join the sentences together into a paragraph-like structure, aka join the list of sentences into one paragraph
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()

        corrected_sentence_chunk = ""
        for i in range(len(joined_sentence_chunk)):
            if i > 0 and joined_sentence_chunk[i].isupper() and joined_sentence_chunk[i-1] == '.':
                corrected_sentence_chunk += ' ' + joined_sentence_chunk[i]
            else:
                corrected_sentence_chunk += joined_sentence_chunk[i]
        chunk_dict["sentence_chunk"] = joined_sentence_chunk

        # Get some stats on our chunks
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 chars

        pages_and_chunks.append(chunk_dict) 

len(pages_and_chunks)

sampled_pages = random.sample(pages_and_chunks, k=1)
print(sampled_pages)

df = pd.DataFrame(pages_and_chunks)
df.describe().round(2)
df.head()

#Filter chunks of text for short chunks
# Get the filtered DataFrame
min_token_length = 30
filtered_df = df[df["chunk_token_count"] <= min_token_length]

# Determine the maximum number of rows to sample (can't sample more than the number of rows in the DataFrame)
num_rows_to_sample = min(5, len(filtered_df))

# Sample the rows
if num_rows_to_sample > 0:
    for row in filtered_df.sample(num_rows_to_sample).iterrows():
        print(f'Chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentence_chunk"]}')
else:
    print("[INFO] No chunks with token count less than or equal to the minimum token length.")

# Filter our DataFrame for rows with under 30 tokens
pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
pages_and_chunks_over_min_token_len[:2]
random.sample(pages_and_chunks_over_min_token_len, k=1)


#Embeddings
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                      device="cpu")


# Create a list of sentences
sentences = ["The Sentence Transformer library provides an easy way to create embeddings.",
             "Sentences can be embedded one by one or in a list.",
             "I like dogs!"]

# Sentences are encoded/embedded by calling model.encode()
embeddings = embedding_model.encode(sentences)
embeddings_dict = dict(zip(sentences, embeddings))

# See the embeddings
for sentence, embedding in embeddings_dict.items():
    print(f"Sentence: {sentence}")
    print(f"Embedding: {embedding}")
    print("")


for item in tqdm(pages_and_chunks_over_min_token_len):
    item["embedding"] = embedding_model.encode(item["sentence_chunk"])
text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]
text_chunks[30]

len(text_chunks)

# Embed all texts in batches
text_chunk_embeddings = embedding_model.encode(text_chunks,
                                               batch_size=32, # you can experiment to find which batch size leads to best results
                                               convert_to_tensor=True)
text_chunk_embeddings
# Check the number of embeddings generated
print(f"Number of embeddings generated: {len(text_chunk_embeddings)}")

# Print the first embedding to see if it looks correct
print(f"Sample embedding: {text_chunk_embeddings[0]}")

pages_and_chunks_over_min_token_len[40]

# Save embeddings to file
text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

text_chunks_and_embedding_df_load = pd.read_csv(embeddings_df_save_path)
text_chunks_and_embedding_df_load.head()


#------------------------------------------RAG beginning-------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# Import texts and embedding df
text_chunks_and_embedding_df = pd.read_csv("text_chunks_and_embeddings_df.csv")

# Convert embedding column back to np.array (it got converted to string when it saved to CSV)
text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

# Convert our embeddings into a torch.tensor
embeddings = torch.tensor(np.stack(text_chunks_and_embedding_df["embedding"].tolist(), axis=0), dtype=torch.float32).to(device)

# Convert texts and embedding df to list of dicts
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

print(text_chunks_and_embedding_df)

print(embeddings.shape)

# Create model
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                      device=device)


# Define the query
query = "play with your dog"
print(f"Query: {query}")

# Embed the query
# Note: it's import to embed you query with the same model you embedding your passages
query_embedding = embedding_model.encode(query, convert_to_tensor=True)

start_time = timer()
dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0]
end_time = timer() 

print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

# Get the top-k results (we'll keep top 5)
top_results_dot_product = torch.topk(dot_scores, k=5)
print(top_results_dot_product)

# Creating an index for faster searching through embeddings
def print_wrapped(text, wrap_length=40):
    wrapped_text = textwrap3.fill(text, wrap_length)
    print(wrapped_text)

print("Results:")
# Loop through zipped together scores and indices from torch.topk
for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
    print(f"Score: {score:.4f}")
    print("Text:")
    print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
    print(f"Page number: {pages_and_chunks[idx]['page_number']}")
    print("\n")

# Define the dot product
def dot_product(vector1, vector2):
    return torch.dot(vector1, vector2)

def cosine_similarity(vector1, vector2):
    dot_product = torch.dot(vector1, vector2)

    # Get Euclidean/L2 norm
    norm_vector1 = torch.sqrt(torch.sum(vector1**2))
    norm_vector2 = torch.sqrt(torch.sum(vector2**2))

    return dot_product / (norm_vector1 * norm_vector2)

# Example vectors/tensors
vector1 = torch.tensor([1, 2, 3], dtype=torch.float32)
vector2 = torch.tensor([1, 2, 3], dtype=torch.float32)
vector3 = torch.tensor([4, 5, 6], dtype=torch.float32)
vector4 = torch.tensor([-1, -2, -3], dtype=torch.float32)

# Calculate dot product
print("Dot product between vector1 and vector2:", dot_product(vector1, vector2))
print("Dot product between vector1 and vector3:", dot_product(vector1, vector3))
print("Dot product between vector1 and vector4:", dot_product(vector1, vector4))

# Cosine similarity
print("Cosine similarity between vector1 and vector2:", cosine_similarity(vector1, vector2))
print("Cosine similarity between vector1 and vector3:", cosine_similarity(vector1, vector3))
print("Cosine similarity between vector1 and vector4:", cosine_similarity(vector1, vector4))

# Giving functionality to semantic search pipeline
def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer=embedding_model,
                                n_resources_to_return: int=5,
                                print_time: bool=True):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Embed the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Get dot product scores on embeddings
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()

    if print_time:
        print(f"[INFO] Time taken to get scores on ({len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

    scores, indices = torch.topk(input=dot_scores,
                                 k=n_resources_to_return)

    return scores, indices

def print_top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks: list[dict]=pages_and_chunks,
                                 n_resources_to_return: int=5):
    """
    Finds relevant passages given a query and prints them out along with their scores.
    """
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)

    # Loop through zipped together scores and indices from torch.topk
    for score, idx in zip(scores, indices):
        print(f"Score: {score:.4f}")
        print("Text:")
        print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
        print(f"Page number: {pages_and_chunks[idx]['page_number']}")
        print("\n")
        

query="choosing a breed right for my family "
retrieve_relevant_resources(query=query, embeddings=embeddings) 
print_top_results_and_scores(query=query, embeddings=embeddings)


#------------------------Loading LLM locally---------------------------------- 

use_quantization_config = True
model_id = "C:\\Users\\SIN8CLJ\\Desktop\\gemma-2b-it"

quatnization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_compute_dtype=torch.float32)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
attn_implementation = "eager"
config = AutoConfig.from_pretrained(model_id)  # Load model config

llm_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path = model_id,
    config = config,
    torch_dtype = torch.float32,  # Use float32 for CPU computations
    low_cpu_mem_usage = True,  # Optimize for low memory usage on CPU
    attn_implementation = attn_implementation,  # Safe attention mechanism for CPU
    trust_remote_code = True  # Enable if you're loading a custom model with remote code
)

#quatnization_config

def get_model_mem_size(model: torch.nn.Module):

    # Get model parameters and buffer sizes
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

    # Calculate various model sizes
    model_mem_bytes = mem_params + mem_buffers # in bytes
    #model_mem_mb = model_mem_bytes / (1024**2) # in megabytes
    model_mem_gb = model_mem_bytes / (1024**3) # in gigabytes

    return {#"model_mem_bytes": model_mem_bytes,
            #"model_mem_mb": round(model_mem_mb, 2),
            "model_mem_gb": round(model_mem_gb, 2)}

print(get_model_mem_size(llm_model))


# Generating text with LLM
input_text = "What can you tell me about dogs?"
print(f"Input text:\n{input_text}")


# Tokenize the input text (turn it into numbers) and send it to the CPU
input_ids = tokenizer(input_text,
                      return_tensors="pt").to("cpu")
llm_model.to("cpu")


# Generate outputs from local LLM
outputs = llm_model.generate(**input_ids,
                             max_new_tokens=256)
print(f"Model output (tokens):\n{outputs[0]}\n")

# Decode the output tokens to text
outputs_decoded = tokenizer.decode(outputs[0])
print(f"Model output (decoded):\n{outputs_decoded}\n")

gemma_questions = [
    "Tell me about the suburban puppies",
    "How do I encurage my dog to chew only chewtoys?",
    "Ho does the tone of my voice influence the puppy?",
    "How to teach my dog to use an outdoor toilet?",
    "When is my birthday?"
]


query_list = gemma_questions

query = random.choice(query_list)
query =  "How to teach my dog to use an outdoor toilet?"

print(f"Query: {query}")

# Get just the scores and indices of top related results
scores, indices = retrieve_relevant_resources(query=query,
                                              embeddings=embeddings)
scores, indices

def prompt_formatter(query: str, 
                     context_items: list[dict]) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    base_prompt = """Based on the following context items about dogs, please answer the query.
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Don't return the thinking, only return the answer.
    Make sure your answers are as explanatory as possible.
    Use the following examples as reference for the ideal answer style.
    \nExample 1:
    Query: How do I choose the right breed of dog for my family?
    Answer: Choosing the right breed involves considering factors such as the size of your living space, your family’s activity level, and the dog's temperament. For example, larger breeds like Labradors may require more space and exercise, while smaller breeds like French Bulldogs are more adaptable to apartment living. Additionally, some breeds are better suited for families with children, such as Golden Retrievers, known for their gentle nature. It's essential to research the energy levels and care requirements of different breeds to ensure they align with your lifestyle.
    \nExample 2:
    Query: What are the key vaccinations puppies need?
    Answer: Puppies require several essential vaccinations to protect them from common diseases. The core vaccines include distemper, parvovirus, and rabies. Distemper protects against a contagious viral disease affecting the respiratory and nervous systems. Parvovirus is a highly contagious virus that attacks the gastrointestinal tract, and rabies is a fatal virus affecting the central nervous system. Most puppies receive these vaccinations in a series, starting at around 6-8 weeks of age, with booster shots following over the next several months.
    \nExample 3:
    Query: How should I train a new puppy for housebreaking?
    Answer: Housebreaking a puppy involves consistency, patience, and positive reinforcement. Begin by establishing a regular routine for feeding, walks, and bathroom breaks. Take your puppy outside frequently, especially after meals, naps, and playtime. Reward your puppy with praise or a treat when they relieve themselves in the appropriate spot. Accidents are normal, so avoid punishment and instead focus on reinforcing the desired behavior. Crate training can also be effective, as dogs naturally avoid soiling their sleeping area.
    \nNow use the following context items to answer the user query:
    {context}
    \nRelevant passages: <extract relevant passages from the context here>
    User query: {query}
    Answer:"""

    # Update base prompt with context items and query   
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt

query = random.choice(query_list)
print(f"Query: {query}")

# Get relevant resources
scores, indices = retrieve_relevant_resources(query=query,
                                              embeddings=embeddings)
    
# Create a list of context items
context_items = [pages_and_chunks[i] for i in indices]

# Format prompt with context items
prompt = prompt_formatter(query=query,
                          context_items=context_items)
print(prompt)

input_ids = tokenizer(prompt, return_tensors="pt")

# Generate an output of tokens
outputs = llm_model.generate(**input_ids,
                             temperature=0.7, # lower temperature = more deterministic outputs, higher temperature = more creative outputs
                             do_sample=True, # whether or not to use sampling, see https://huyenchip.com/2024/01/16/sampling.html for more
                             max_new_tokens=256) # how many new tokens to generate from prompt 

# Turn the output tokens into text
output_text = tokenizer.decode(outputs[0])

print(f"Query: {query}")
print(f"RAG answer:\n{output_text.replace(prompt, '')}")

def ask(query, 
        temperature=0.7,
        max_new_tokens=256,
        format_answer_text=True, 
        return_answer_only=True):
    """
    Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
    """

  # Get just the scores and indices of top related results
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings)
    
    # Create a list of context items
    context_items = [pages_and_chunks[i] for i in indices]

    # Add score to context item
    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu() # return score back to CPU 
        
    # Format the prompt with context items
    prompt = prompt_formatter(query=query,
                              context_items=context_items)
    
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt")

    # Generate an output of tokens
    outputs = llm_model.generate(**input_ids,
                                 temperature=temperature,
                                 do_sample=True,
                                 max_new_tokens=max_new_tokens)
    
    # Turn the output tokens into text
    output_text = tokenizer.decode(outputs[0])

    if format_answer_text:
        # Replace special tokens and unnecessary help message
        output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "")

    # Only return the answer without the context items
    if return_answer_only:
        return output_text
    
    return output_text, context_items

query = random.choice(query_list)
print(f"Query: {query}")

# Answer query with context and return context 
answer, context_items = ask(query=query, 
                            temperature=0.7,
                            max_new_tokens=81,
                            return_answer_only=False)

print(f"Answer:\n")
print_wrapped(answer)
print(f"Context items:")
print(context_items)