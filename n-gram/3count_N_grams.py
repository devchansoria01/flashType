from collections import Counter
import pickle
import os

# --- CONFIGURATION ---
INPUT_NGRAMS_PATH = '2tokenization_Output.pkl'
OUTPUT_COUNTS_PATH = '3ngram_counts.pkl'
# ---------------------

print("Starting Step 3: Counting N-grams for all levels...")

if not os.path.exists(INPUT_NGRAMS_PATH):
    print(f"Error: N-grams file not found at {INPUT_NGRAMS_PATH}. Run step 2_tokenize.py first.")
    exit()

# 1. Load the N-grams generated in Step 2
with open(INPUT_NGRAMS_PATH, 'rb') as f:
    data = pickle.load(f)

MAX_N = data['MAX_N']
all_ngrams_data = data['ngrams']
context_ngrams_data = data['context_ngrams']

counts_to_save = {'MAX_N': MAX_N, 'ngram_counts': {}, 'context_counts': {}}

# 2. Calculate Counts for N=2, 3, 4
for N_size in range(2, MAX_N + 1):
    print(f"Counting N={N_size} grams...")
    
    ngram_counts = Counter(all_ngrams_data[N_size])
    context_counts = Counter(context_ngrams_data[N_size])
    
    counts_to_save['ngram_counts'][N_size] = ngram_counts
    counts_to_save['context_counts'][N_size] = context_counts
    
    print(f"  - Unique N={N_size} grams: {len(ngram_counts)}")

# 3. Save the counts
with open(OUTPUT_COUNTS_PATH, 'wb') as f:
    pickle.dump(counts_to_save, f)

print(f"\n--- Counting Complete ---")
print(f"All N-gram counts saved to: {OUTPUT_COUNTS_PATH}")