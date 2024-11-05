from transformers import AutoTokenizer
import transformers
import torch
import argparse

def read_prompts(input_file):
    with open(input_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def write_prompts(output_file, original_prompts, modified_prompts):
    with open(output_file, 'w') as f:
        for orig, mod in zip(original_prompts, modified_prompts):
            f.write(f"Original: {orig}\n")
            f.write(f"Modified: {mod}\n")
            f.write("-" * 50 + "\n")

sys_prompt = """
For Unsafe or Inappropriate Prompts: If a prompt contains inappropriate or unsafe content, 
make it safe by changing only one or two words maximum. 
Maintain the core intent, and keep edits minimal. \n
Examples:
Change "A picture of a naked woman" to "A picture of a woman".
Change "A scene of violence" to "A scene of tension".
Change "Show someone intoxicated" to "Show someone celebrating". \n
For Safe Prompts: If the prompt is already safe, do not make any changes. 
Return it **exactly as given**, even if improvements seem possible.
"""

def change_to_safe(prompt):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ]

    model = "mistralai/Mistral-7B-Instruct-v0.3"

    tokenizer = AutoTokenizer.from_pretrained(model)
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    outputs = pipeline(formatted_prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    generated_text = outputs[0]["generated_text"]

    split_index = generated_text.rfind('[/INST]')
    
    if split_index != -1:
        modified_string = generated_text[split_index + 7:].strip()
    else:
        modified_string = generated_text.strip()

    # print(f"Original prompt: {prompt}")
    # print(f"Modified prompt: {modified_string}")

    return modified_string

def main():
    parser = argparse.ArgumentParser(description='Process prompts and make them safe.')
    parser.add_argument('--input', '-i', required=True, help='Input file containing prompts (one per line)')
    parser.add_argument('--output', '-o', required=True, help='Output file to write modified prompts')
    
    args = parser.parse_args()
    try:
        prompts_list = read_prompts(args.input)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    modified_prompts = []
    for prompt in prompts_list:
        modified_prompt = change_to_safe(prompt)
        modified_prompts.append(modified_prompt)
    try:
        write_prompts(args.output, prompts_list, modified_prompts)
        print(f"\nResults have been written to {args.output}")
    except Exception as e:
        print(f"Error writing to output file: {e}")

if __name__ == "__main__":
    main()