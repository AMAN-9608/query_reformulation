# query_reformulation

## Overview

The **Warmup API** and **Warmup Frontend** are components of a Streamlit and FastAPI application designed to fetch and rank stories from Hacker News based on a user's bio. I've utilized sentence embeddings to rank stories similar to the user's bio.

## Project Structure

- **fastapi_server.py**: Contains the backend logic for fetching stories from Hacker News and processing user input.
- **warmup_frontend.py**: The streamlit frontend that interacts with the user, collects input, and displays ranked stories.
- **warmup_app.py**: Streamlit app that is used for hosting on streamlit cloud.

## Local Installation
To run the app locally, simply execute "bash start.sh" in your terminal, which will install the requirements, and run the flask and streamlit files. You can then proceed to the localhost streamlit address to access the application, which will in turn call the API using POST method to return results.

## Design Decisions

* For the machine learning model that reformulates the input, below are the design decisions taken:
    * To select the pretrained large language model for fine tuning, since I was working on my local machine with a GPU vRAM of 6GB, I had to go for a model with relatively fewer parameters. I chose the google flan t5 small model with 77 million parameters, since it has been trained on a much broader set of tasks. I tried training on the t5 tiny model as well, but the results were hardly passable.
    * To select the relevant data, I used the MS-MARCO query reformulation diamond dataset (https://github.com/Narabzad/msmarco-query-reformulation/tree/main/datasets)
    * Some of the parameters used during fine-tuning:
        *
    * I also tried various prompts to further tune the model, some of which were:
        * No prompt
        * "Reformulate the query: "
        * "Given the following sentence generate a search engine query that will be able to obtain requested information: "
        * "Question: Given the following input sentence generate a search engine query 
        that will be able to obtain requested information: "
    * Given the ask of the task that the model return multiply queries, the  
    top_k This parameter limits the number of highest probability vocabulary tokens to consider during generation. For example, if top_k=50, only the top 50 tokens with the highest probabilities will be considered for the next token generation.
    num_return_sequences Specifies the number of different sequences to generate for each input. In this case, it is set to 5, meaning the model will return five different generated outputs.
    the output_score is returned for each sequence, and we set a threshold than is # Returning variable number of sequences based on threshold of scores, less negative means higher likelihood acc to model
    The output_scores are a series of log probabilities associated with each token in the generated sequences. These scores represent the model's confidence level in selecting each token during the generation process.
    Closer to 0: Higher probability, higher confidence (e.g., -0.1).
More Negative: Lower probability, lower confidence (e.g., -5.0)
    

* For the API, the design decisions taken were:
    * At first I thought of using a flask server api for hosting the ml inference api endpoint. However, for 8 future scalability and just the fact that fastAPI is faster than Flask due to its async capabilities.
    * Initially I tried using the trained model as is, but using dynamic quantization on the model sped up the response time of the API from ~250ms to ~150ms. I quantized the linear layers and used dtype=qint8.
    * To further try to reduce the time to first byte, I also tried to convert the model to onnx format. But that did not yield any reduction in response time either.