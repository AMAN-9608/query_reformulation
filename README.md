# query_reformulation

## Overview

The **fastapi_server.py** and **streamlit_frontend.py** files are components of a Streamlit and FastAPI application designed to reformulate a given user query. The **streamlit_app.py** file is used to host the live demo on streamlit cloud.

## Project Structure

- **fastapi_server.py**: Hosts the ML model inference endpoint that is called by the streamlit frontend.
- **streamlit_frontend.py**: The streamlit frontend that interacts with the user, collects input query, makes a call to the API and returns reformulated query/queries.
- **streamlit_app.py**: Streamlit app that is used for hosting on streamlit cloud. Does not contain any api calls.
- **train.py**: python file that contains code for training the ML model.

## Local Installation
To run the app locally, execute the following commands:

* pip install -r requirements.txt
* uvicorn fastapi_server:app --workers 8
* streamlit run streamlit_frontend.py

## Design Decisions

* For the machine learning model that reformulates the input query, below are the design decisions taken:
    * To select the pretrained large language model for fine tuning, since I was working on my local machine with a GPU vRAM of 6GB, I had to go for a model with relatively fewer parameters. I chose the google flan t5 small model with 77 million parameters, since it has been trained on a much broader set of tasks. I tried training on the t5 tiny model as well, but the results were hardly passable.
    * To select the relevant data, I used the MS-MARCO query reformulation diamond dataset found at:<br> https://github.com/Narabzad/msmarco-query-reformulation/tree/main/datasets <br> This dataset has around 188k training examples.
    * Some of the parameters used during fine-tuning:
        * batch size = 5
        * max sequence length = 50
        * num_epochs = 1
        * learning rate = 1e-5
    * I also tried various prompts to further tune the model, some of which were:
        * No prompt
        * "Reformulate the query: "
        * "Given the following sentence generate a search engine query that will be able to obtain requested information: "
        * "Question: Given the following input sentence generate a search engine query 
        that will be able to obtain requested information: "
    * Given the ask of the task that the model return multiply queries, I chose to achieve this via
        * num_return_sequences = 5, this parameter specifies the number of different sequences to generate for each input. 
        * output_score is returned for each sequence, where less negative (closer to 0) means higher likelihood according to model. The output_scores are a series of log probabilities associated with each token in the generated sequences. These scores represent the model's confidence level in selecting each token during the generation process.
        * threshold = -2 was applied to the score of each generated sequence, meaning that any sequence with a score <-2 was rejected and not returned.
    

* For the API, the design decisions taken were:
    * At first I thought of using a flask server api for hosting the ml inference api endpoint. However, for future scalability and just the fact that fastAPI is faster than Flask due to its async capabilities, I went ahead with fastAPI.
    * Initially I tried using the trained model as is, but using dynamic quantization on the model sped up the response time of the API from ~250ms to ~150ms. I quantized the linear layers and used dtype=qint8.
    * To further try to reduce the time to first byte, I also tried to convert the model to onnx format. But that did not yield any reduction in response time either.


## Testing and Scope for Improvement
Using Postman for testing my API locally, I was able to get a response time of around 150ms, and couldn't get it below 100ms. 95% of that time was spent waiting for the first byte, meaning that there is still some scope for improving model inference speed.

Some of the ways to decrease the response time and the model output that are worth exploring are:
* Increase the number of training epochs for the data
* Using a vector db to match returned sequences against input, and return only those with a high similarity
* Implement streaming response of some sort that can return the response in chunks
