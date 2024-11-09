import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

st.title("Query Reformulation App")

@st.cache_resource 
def load_model():
    device = torch.device("cpu")  # Use CPU for the quantized model
    tokenizer = T5Tokenizer.from_pretrained("aman9608/query_reformulation")
    model = T5ForConditionalGeneration.from_pretrained("aman9608/query_reformulation").to(device)
    
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.eval() 
    return model, tokenizer, device

model, tokenizer, device= load_model()

prompt = """Question: Given the following input text, reformulate the query
to make it more specific, clear, and contextually appropriate for retrieving
relevant information: """
prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

input_text = st.text_area("Enter your query:", "")

if st.button("Reformulate Query"):
    if input_text:
        input_ids = torch.cat(
            [prompt_tokens, tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)],
            dim=-1
        )

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                num_return_sequences=5,
                top_k=50,
                top_p=0.95,
                temperature=1,
                num_beams=5,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

            threshold = -2
            filtered_outputs = []

            for i, output in enumerate(outputs.sequences):
                score = outputs.sequences_scores[i].item()
                if score > threshold:
                    filtered_outputs.append(tokenizer.decode(output, skip_special_tokens=True))

            if filtered_outputs:
                st.success("Reformulated Queries:")
                for idx, query in enumerate(filtered_outputs):
                    st.write(f"{idx + 1}: {query}")
            else:
                st.warning("No valid reformulated queries found.")
    else:
        st.warning("Please enter a query.")