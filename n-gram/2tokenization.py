import nltk
from nltk.util import ngrams
import pickle
import os

# --- CONFIGURATION ---
INPUT_FILE_PATH = '1combined_cleaned_messages.txt'
OUTPUT_NGRAMS_PATH = '2tokenization_Output.pkl'
MAX_N = 4 # Start with N=4
# ---------------------

print(f"Starting Step 2: Tokenization for N=2, N=3, and N={MAX_N}...")

if not os.path.exists(INPUT_FILE_PATH):
    print(f"Error: Corpus file not found at {INPUT_FILE_PATH}.")
    exit()

try:
    with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
        corpus = f.read().lower() 
except Exception as e:
    print(f"Error reading corpus file: {e}")
    exit()

tokens = nltk.word_tokenize(corpus)
data_to_save = {'MAX_N': MAX_N, 'ngrams': {}, 'context_ngrams': {}}

# Loop from N=2 up to MAX_N (4)
for N_size in range(2, MAX_N + 1):
    print(f"Generating N={N_size} grams...")
    
    # 1. Prepare tokens with Start markers for this N_size
    start_tokens = ['<s>'] * (N_size - 1)
    padded_tokens = start_tokens + tokens

    # 2. Generate N-grams (sequences of length N)
    all_ngrams = list(ngrams(padded_tokens, N_size))

    # 3. Generate Context N-grams (sequences of length N-1)
    context_ngrams = list(ngrams(padded_tokens, N_size - 1))

    data_to_save['ngrams'][N_size] = all_ngrams
    data_to_save['context_ngrams'][N_size] = context_ngrams

print(f"\n--- Tokenization Complete ---")
print(f"Token data for N=2, 3, 4 saved to: {OUTPUT_NGRAMS_PATH}")

# 4. Save the generated lists
with open(OUTPUT_NGRAMS_PATH, 'wb') as f:
    pickle.dump(data_to_save, f)