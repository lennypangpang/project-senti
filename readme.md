# Project Sentiment

Project Sentiment is a project involving analyzing annotated financial news datasets

## Further Info
This project includes a docker .devcontainer file with links to a tensorflow docker image to avoid any package dependencies when using NVIDIA GPU/CUDA to accelerate model training.

## Model Details

The python notebook(s) presents three models: 
1. Traditional LSTM + Pre-trained GloVe embeddings for embedding layer (1)
2. Traditional LSTM + Learned embeddings for embedding layer (2)
3. BiDirectional LSTM + Attention layer from keras (Dot Product Attention) (3)

Each .py file includes the models, while the python notebooks shows the results

## What's next?

In this section, when discussing a specific model, I will refer to the number specificed in the Model Details section in this readme file. 
1. When training the LSTM w/Pre-Trained GloVe embeddings, we used a global GloVe embedding. Ideally, we should move to use a embeddings finetuned with financial data - so the model would understand financial jargon better. 

2. ~3000 financial headlines isn't that much, adding more data may improve model performance

3. 


## References


