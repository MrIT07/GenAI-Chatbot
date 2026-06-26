!pip install transformers langchain torch gradio

!pip uninstall -y langchain
!pip install langchain==0.1.20

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.memory import ConversationBufferMemory
import gradio as gr

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# Check if CUDA is available and move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer with token
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to generate chatbot response without chat history
def chatbot_response(user_input):

    # Chat template (required for Qwen Instruct models)
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": user_input},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize the formatted prompt
    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(device)

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.15,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    # Remove prompt tokens from output
    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]

    # Decode only the generated answer
    bot_reply = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return bot_reply.strip()

chatbot_response("Hi how are you?")

def chatbot_interface(user_input, history):
    response = chatbot_response(user_input)
    history.append((user_input, response))
    return history, ""

# Creating the Gradio Interface
demo = gr.Blocks()

with demo:
    gr.Markdown("## GenAI Chatbot")
    chatbot = gr.Chatbot()
    with gr.Row():
        user_input = gr.Textbox(placeholder="Type your message here...", lines=1)
        send_button = gr.Button("Send")

    history = gr.State([])

    send_button.click(chatbot_interface, inputs=[user_input, history], outputs=[chatbot, user_input])
    user_input.submit(chatbot_interface, inputs=[user_input, history], outputs=[chatbot, user_input])

# Launch the chatbot
demo.launch(share = True)
