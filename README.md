## Tips
I've used mini BERT with an embedding size of `256`. You probably need to configure the LTR model to use the embedding size of your own version of BERT, i.e. `512`.
Put your own BERT model in `models` folder. Update the address of models in notebooks accordingly.

## Training
1. First of all, run `prep_data.ipynb`. This notebook calculates the embedding for queries and the candidate expansion terms. If you want to change how the embeddings are calculated, you should edit the `get_query_and_candidates_embeddings` in the `utils.py` folder.

   - `prep_data.ipynb` will create a pickle file in `data` folder which includes a dictionary mapping `new_q`s to embedding tensors. This pickle file is loaded and given to `RankNetDatasetWithMap` class to prepare our training dataset.

 2. after that, run `train.ipynb` notebook to train the model.



## Evaluation
You can find an example of evaluation in `rewarder_example.ipynb` notebook.

You must give the `cord_qrels.json` file to an instance of the `Rewarder` class. Then using the `reward_one` method of this class you can calculate the result for the given `metric` at `n_results`.

For example the following instantition of this class will evaluate the `recall@100` of the `coronavirus origin asia` 

```python
rewarder = Rewarder(
        "./data/cord_qrels.json", n_results=at_n, metric=metric
    )
rewarder.reward_one('coronavirus origin asia', 'coronavirus origin')
 # the second input is the id of query, 
 # so the evalution is based on the related papers for this query in qrels.json file 
 # (here id is the base query itself)
```