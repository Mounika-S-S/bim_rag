import os
import json
import glob

def chat_to_chatml(history_dir="data/chat_history", output_file="data/processed/finetune_dataset.jsonl"):
    """
    Parses chat history JSON files and formats them into ChatML structured JSONL for HuggingFace `trl`.
    Format: {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed", exist_ok=True)

    json_files = glob.glob(os.path.join(history_dir, "*.json"))
    
    if not json_files:
        print(f"No chat history files found in {history_dir}")
        return

    system_prompt = (
        "You are an expert BIM compliance assistant. Answer questions based on the provided context from BIM layers (L1-L5)."
    )

    dataset = []

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                chat_session = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping {file_path} - Invalid JSON")
                continue
            
            # Group into conversational pairs
            messages = [{"role": "system", "content": system_prompt}]
            
            for msg in chat_session:
                role = msg.get("role")
                content = msg.get("text", "")
                
                if role and content:
                    messages.append({"role": role, "content": content})
            
            # Append if we have at least user and assistant
            if len(messages) >= 3:
                dataset.append({"messages": messages})

    with open(output_file, "w", encoding="utf-8") as out_f:
        for data in dataset:
            out_f.write(json.dumps(data) + "\n")
            
    print(f"Successfully converted {len(dataset)} chat sessions into ChatML format.")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    chat_to_chatml()
