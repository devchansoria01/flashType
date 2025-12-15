import nltk
import pickle
import random
import re
import os

# --- CONFIGURATION ---
INPUT_MODEL_PATH = '4output.pkl'
# ---------------------

print("Starting Step 5: Real-Time Prediction Tool...")

if not os.path.exists(INPUT_MODEL_PATH):
    print(f"Error: Model file not found at {INPUT_MODEL_PATH}. Run the training script (Step 4) first.")
    exit()

# 1. Load the trained multi-level model ONCE
with open(INPUT_MODEL_PATH, 'rb') as f:
    multi_level_model = pickle.load(f)

MAX_N = multi_level_model['MAX_N']
MODELS = multi_level_model['models']
print(f"Successfully loaded models from N=2 to N={MAX_N}.")
print("-" * 50)
print("Real-Time Predictor Ready. Type a phrase and press Enter.")
print("Type 'quit' or 'exit' to stop.")
print("-" * 50)


def predict_next_word(input_text, all_models, max_n):
    """
    Uses Backoff Smoothing: tries prediction with max_n, then max_n-1, and so on.
    (This function is unchanged from the previous step)
    """
    # Prepare the input: lowercase, tokenize, and clean punctuation
    input_text = re.sub(r'[.,!?;:()"]', r'', input_text)
    input_tokens = nltk.word_tokenize(input_text.lower())
    
    # Iterate from the largest N down to N=2 (Bigram)
    for N_size in range(max_n, 1, -1):
        
        model = all_models.get(N_size)
        if not model:
            continue
            
        required_context_length = N_size - 1
        
        # --- Prepare Context Tokens ---
        if len(input_tokens) < required_context_length:
            context_tokens = ['<s>'] * (required_context_length - len(input_tokens)) + input_tokens
        else:
            context_tokens = input_tokens[-required_context_length:]
        
        context_tuple = tuple(context_tokens)

        # Look up the context in the current N-gram model
        if context_tuple in model:
            print(f"\n[Using N={N_size} Model]")
            
            predictions = model[context_tuple]
            
            words = [item[0] for item in predictions]
            probabilities = [item[1] for item in predictions]
            
            # --- Output Formatting for Real-time ---
            # Randomly select one word based on probability
            predicted_word = random.choices(words, weights=probabilities, k=1)[0]
            
            # Show the top 3 possibilities
            top_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:3]
            
            
            print(f"Top Suggestions:")
            # Displaying suggestions on one line for better UI
            suggestions = [f"{w} ({p:.2f})" for w, p in top_predictions]
            print(f"  > {' | '.join(suggestions)}")
            
            return predicted_word

    # If the loop finishes without finding any prediction (even in N=2)
    return "[No common sequence found in chat history.]"


# --- REAL-TIME INPUT LOOP (The core change) ---

while True:
    try:
        user_input = input("\nYour Phrase > ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Exiting real-time predictor. Goodbye!")
            break
            
        if not user_input:
            continue

        # Get the prediction using the loaded models
        predicted_word = predict_next_word(user_input, MODELS, MAX_N)
        
        # The final output:
        print(f"\nPrediction for '{user_input}': {user_input} \033[1m{predicted_word}\033[0m")
        
    except KeyboardInterrupt:
        # Allows user to stop with Ctrl+C
        print("\nExiting real-time predictor. Goodbye!")
        break
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        break