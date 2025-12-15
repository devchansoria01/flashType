import pickle
import os

# --- CONFIGURATION ---
INPUT_COUNTS_PATH = '3ngram_counts.pkl'
OUTPUT_MODEL_PATH = '4output.pkl'
# ---------------------

print("Starting Step 4: Probability Calculation and Model Saving...")

if not os.path.exists(INPUT_COUNTS_PATH):
    print(f"Error: Counts file not found at {INPUT_COUNTS_PATH}. Run step 3_count_ngrams.py first.")
    exit()

# 1. Load the N-gram counts
with open(INPUT_COUNTS_PATH, 'rb') as f:
    data = pickle.load(f)

MAX_N = data['MAX_N']
ngram_counts_data = data['ngram_counts']
context_counts_data = data['context_counts']

final_probability_models = {'MAX_N': MAX_N, 'models': {}}

# 2. Calculate Conditional Probabilities for N=2, 3, 4
for N_size in range(2, MAX_N + 1):
    print(f"Building Model for N={N_size}...")
    
    ngram_counts = ngram_counts_data[N_size]
    context_counts = context_counts_data[N_size]
    probability_model = {}

    for ngram, count in ngram_counts.items():
        
        context = ngram[:-1]
        word = ngram[-1]
        context_count = context_counts[context]
        
        probability = count / context_count
        
        if context not in probability_model:
            probability_model[context] = []
            
        probability_model[context].append((word, probability))
    
    final_probability_models['models'][N_size] = probability_model
    print(f"  - N={N_size} model trained on {len(probability_model)} contexts.")

# 3. Save the final multi-level model dictionary
with open(OUTPUT_MODEL_PATH, 'wb') as f:
    pickle.dump(final_probability_models, f)

print(f"\n--- Model Creation Complete ---")
print(f"Multi-level model (N=2 to N={MAX_N}) saved to: {OUTPUT_MODEL_PATH}")