# Project Sentiment

Project Sentiment is a project involving analyzing annotated financial news dataset to predict its sentiment. 

## Further Info
This project includes a docker .devcontainer file with links to a tensorflow docker image to avoid any package dependencies when using NVIDIA GPU/CUDA to accelerate model training.

## Model Details

The python notebook(s) presents three models: 
1. Traditional LSTM + Pre-trained GloVe embeddings for embedding layer (1)
2. Traditional LSTM + Learned embeddings for embedding layer (2)
3. BiDirectional LSTM + Attention layer from keras (Dot Product Attention) (3)

Each .py file includes the models, while the python notebooks shows the results

## What's next?

1. Domain-Specific Embeddings: For the LSTM model utilizing pre-trained embeddings, we employed a globally trained GloVe embedding. This approach does not account for domain-specific language. Ideally, embeddings should be fine-tuned on financial corpora to better capture the nuances of financial jargon and improve the model’s contextual understanding.

2. Specialized Polarity Lexicons: Constructing a domain-specific semantic polarity lexicon could further support sentiment analysis tasks. Prior work by Malo, Pekka, and Sinha demonstrates that combining Support Vector Machines with the MPQA subjectivity lexicon yields strong results in financial contexts. For instance, while not a financial example, "not bad" could be considered "fine" whilst in other contexts - "terrible"
   
<img width="464" alt="Screenshot 2025-05-31 at 19 53 56" src="https://github.com/user-attachments/assets/fcd75f33-3ab7-438b-b01a-be8499aa4ee1" />
<img width="736" alt="Screenshot 2025-05-31 at 19 54 08" src="https://github.com/user-attachments/assets/ead4285b-db83-4dd2-a210-141731b77d73" />

3. Exanding Corpora: This dataset is comprised of a relatively modest ~3000 headlines. This relatively limited dataset may constrain performance, and lead to overfitting. 
  
4. Contemporary NLP approaches like sentence-transformers with an attached multi-layered perception and fine tuned models such as FinBERT may warrant further investigation. These contemporary methods may capture more nuances in Financial sentiment more effectiverly than traditional LSTM approaches.

5. Clickbait? Headlines may not include all polarity and semantic information to capture the full sentiment of an article. Hence, in future models it may be worth including the body. 

## References
Malo, Pekka & Sinha, Ankur & Takala, Pyry & Korhonen, Pekka & Wallenius, Jyrki. (2014). Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts. Journal of the American Society for Information Science and Technology. 10.1002/asi.23062. 

Peng Wu, Xiaotong Li, Chen Ling, Shengchun Ding, Si Shen, Sentiment classification using attention mechanism and bidirectional long short-term memory network, Applied Soft Computing, Volume 112,2021, 107792, ISSN 1568-4946, https://doi.org/10.1016/j.asoc.2021.107792.


