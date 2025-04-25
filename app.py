import gradio as gr
from huggingface_hub import InferenceClient
import json  # Import the json module
import os 

# Get the token from an environment variable
hf_token = os.getenv("HF_TOKEN")
# Initialize the InferenceClient with the chosen summarization model
client = InferenceClient("facebook/bart-large-cnn", token=hf_token)

def summarize_text(input_text, max_length=150, min_length=30):
    """
    Function to generate a summary from the input text using the Inference API.
    
    Args:
        input_text (str): The text to be summarized.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.
    
    Returns:
        str: The summarized text.
    """
    # Generate the summary using the Inference API
    response = client.post(
        json={
            "inputs": input_text,
            "parameters": {
                "max_length": max_length,
                "min_length": min_length,
                "do_sample": False
            }
        }
    )
    # Decode the raw bytes response into a string
    response_str = response.decode("utf-8")
    # Parse the string as JSON
    result = json.loads(response_str)
    return result[0]['summary_text']

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Text Summarization Tool")
    gr.Markdown("Enter a long document or article below, and get a concise summary instantly!")
    
    with gr.Row():
        input_text = gr.Textbox(
            label="Input Text",
            placeholder="Paste your long text here...",
            lines=10
        )        
    
    with gr.Row():
        max_length = gr.Slider(
            minimum=50,
            maximum=500,
            value=150,
            step=10,
            label="Max Summary Length"
        )
        min_length = gr.Slider(
            minimum=10,
            maximum=200,
            value=30,
            step=10,
            label="Min Summary Length"
        )
    
    summarize_button = gr.Button("Summarize")
    
    output_summary = gr.Textbox(
        label="Summary",
        placeholder="The summary will appear here...",
        lines=5
    )
      
    # Link the button to the summarize function
    summarize_button.click(
        fn=summarize_text,
        inputs=[input_text, max_length, min_length],
        outputs=output_summary
    )

# Launch the app
demo.launch()