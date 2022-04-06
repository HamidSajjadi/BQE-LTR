##Tips
I've used mini BERT with an embedding size of `256`. You probably need to configure the LTR model to use the embedding size of your own version of BERT, i.e. `512`.

## Training
1. First of all, run `prep_data.ipynb`. This notebook calculates the embedding for queries and the candidate expansion terms. If you want to change how the embeddings are calculated, you should edit the `get_query_and_candidates_embeddings` in the `utils.py` folder.

   - `prep_data.ipynb` will create a pickle file in `data` folder which includes a dictionary mapping `new_q`s to embedding tensors. This pickle file is loaded and given to `RankNetDatasetWithMap` class to prepare our training dataset.

 2. after that, run `train.ipynb` notebook to train the model.



## Evaluation
the evaluation notebook will be added shortly.