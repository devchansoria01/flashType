import re
import os 

YOUR_NAME = 'Dev Chansoria' 

FILE_PATHS = [
    'chats/anita.txt', 
    'chats/tanishka.txt', 
    'chats/yash.txt',
    'chats/ritesh.txt',
] 

OUTPUT_FILE_PATH = '1combined_cleaned_messages.txt'

REMOVE_EMOJIS = True 

def clean_whatsapp_chat(input_file, sender_name, remove_emojis): 
    """
    Parses a single WhatsApp chat file, extracts messages from the specified sender, 
    removes metadata, URLs, and optional emojis, and returns a list of cleaned messages.
    """
    
    # Regex Patterns
    CHAT_LINE_PREFIX_PATTERN = r'^.*? - ([^:]+):\s*' 
    SYSTEM_MESSAGE_PATTERN = r'<\S+\s+omitted>'
    URL_PATTERN = r'https?://\S+|www\.\S+|\S+\.\S+/\S+' # For removing web links
    
    # Comprehensive EMOJI_PATTERN 
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U00002702-\U000027B0"  # Dingbats
        "]+"
        "(\u200d\u2640|\u200d\u2642|\ufe0f|\u20e3)?" 
        , flags=re.UNICODE)
    
    my_messages = []
    # Flag to ensure only *your* multi-line messages are appended (fixes date contamination)
    last_sender_was_me = False 
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue 

                match = re.match(CHAT_LINE_PREFIX_PATTERN, line)
                
                if match:
                    sender = match.group(1).strip()
                    
                    if sender == sender_name:
                        # --- Message is from the user (KEEP) ---
                        message_content = re.sub(CHAT_LINE_PREFIX_PATTERN, '', line, 1).strip()
                        
                        if not re.search(SYSTEM_MESSAGE_PATTERN, message_content):
                            
                            # Remove URLs first
                            message_content = re.sub(URL_PATTERN, '', message_content).strip()
                            
                            if remove_emojis:
                                message_content = EMOJI_PATTERN.sub(r'', message_content)
                                message_content = re.sub(r'\s+', ' ', message_content).strip()
                            
                            # Only append if the message is not empty after all cleaning
                            if message_content:
                                my_messages.append(message_content)
                                last_sender_was_me = True 
                    else:
                        # --- Message is from another sender (DISCARD) ---
                        last_sender_was_me = False 
                else:
                    # --- Continuation Line ---
                    # Only append if the last line we kept was from the user
                    if my_messages and last_sender_was_me:
                        
                        # Apply URL and emoji removal to the continuation line
                        cleaned_continuation = re.sub(URL_PATTERN, '', line)
                        if remove_emojis:
                            cleaned_continuation = EMOJI_PATTERN.sub(r'', cleaned_continuation)
                        
                        if cleaned_continuation.strip():
                            my_messages[-1] += " " + cleaned_continuation.strip()
                        
    except FileNotFoundError:
        print(f"⚠️ Warning: File not found at {input_file}. Skipping this file.")
        
    return my_messages 

# --- MAIN EXECUTION: Loop Through Files and Aggregate ---

all_messages = []
total_messages_processed = 0

print(f"Starting analysis for {len(FILE_PATHS)} chat files...")

for file_path in FILE_PATHS:
    if os.path.exists(file_path):
        print(f"Processing file: {file_path}...")
        
        messages_from_file = clean_whatsapp_chat(file_path, YOUR_NAME, REMOVE_EMOJIS)
        
        all_messages.extend(messages_from_file)
        total_messages_processed += len(messages_from_file)
    else:
        print(f"⚠️ Error: File '{file_path}' not found. Check your paths ('chats/' directory).")


# Save the Combined Corpus after all files are processed
if all_messages:
    # Joining with newline (\n) helps NLTK treat each message as a sentence/sequence
    final_corpus = '\n'.join(all_messages) 
    
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f_out:
        f_out.write(final_corpus)

    print("\n--- Cleaning Complete ---")
    print(f"Total messages successfully extracted: {total_messages_processed}")
    print(f"Combined clean corpus saved to: {OUTPUT_FILE_PATH}")
    print("\nNext step: Run '2_train_model.py' using the combined file.")
else:
    print("\n--- Cleaning Failed ---")
    print("Zero messages were extracted. Check your file paths and YOUR_NAME configuration.")