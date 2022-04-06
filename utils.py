import pickle
from typing import List
import torch
import random
import numpy as np

def save_pickle(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_pickle(filename_):
    with open(filename_, 'rb') as f:
        ret_di=pickle.load(f)
    return ret_di


def get_query_and_candidates_embeddings(queries: List[str], candidates: List[str], tokenizer, bert_model, max_length = 8, device = None) -> torch.Tensor:
        """Given a list of queries and candidate tokens, creates embedding of the new query calculated by concating them

        Args:
            queries (List[str])
            candidates (List[str])

        Returns:
            torch.Tensor
        """

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
       
        with torch.no_grad():
            sample_expanded_queries = [q + " " + c for q, c in zip(queries, candidates)]
            
            tokenized_new_q = tokenizer(
                sample_expanded_queries,padding="max_length",
                max_length=max_length ,
                truncation=True,
                return_tensors="pt",
                
            )

            tokenized_new_q = {k: v.to(device) for k, v in tokenized_new_q.items()}
            out = bert_model(**tokenized_new_q)
            emb = out[0]
            return emb
            
    
def set_seed(number = 42):
    torch.manual_seed(number)
    np.random.seed(number)
    random.seed(number)

