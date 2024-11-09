from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import torch

app = FastAPI()

model = None
tokenizer = None
device = None
startup_complete = False
prompt_tokens = None

class InputText(BaseModel):
    query: str


@app.on_event("startup")
async def startup_event():
    print("The API Server is starting up!")
    global model, tokenizer, device, startup_complete, prompt_tokens

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # for quantized model

    tokenizer = T5Tokenizer.from_pretrained("aman9608/query_reformulation")
    model = T5ForConditionalGeneration.from_pretrained(
        "aman9608/query_reformulation"
    ).to(device)

    # Apply dynamic quantization
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # model.half() # Uncomment if using pytorch model to convert to fp16
    model.eval()

    # To Load ONNX model
    # onnx_translation = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    
    prompt = """Question: Given the following input text, reformulate the query
    to make it more specific, clear, and contextually appropriate for retrieving
    relevant information: """

    prompt_tokens = tokenizer(
        prompt, return_tensors="pt", padding=True, truncation=True
    ).input_ids.to(device)

    with torch.inference_mode(): 
        model.generate(prompt_tokens)
    
    startup_complete = True
    print("The API Server is live!")


@app.get("/health")
async def health_check():
    if startup_complete:
        return {"status": "ready"}
    else:
        return {"status": "starting up"}


@app.post("/reformulate_query", response_model=list[str])
async def reformulate_query(input_text: InputText):
    input_ids = torch.cat(
    [prompt_tokens, tokenizer(input_text.query, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)],
    dim=-1)

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

        return filtered_outputs

        # return onnx_translation(f"Reword the input into a search engine query: {input_text}")
