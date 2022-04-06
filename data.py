import pandas as pd
from typing import List, Dict
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch
from utils import get_query_and_candidates_embeddings



class RankNetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, bert: BertModel, device=None, max_length=8) -> None:

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.df = df
        self.tokenizer = tokenizer
        self.bert_model = bert
        self.max_length = max_length
    
    def _get_query_and_candidates_embeddings(self, queries: List[str], candidates: List[str]) -> torch.Tensor:
        """Given a list of queries and candidate tokens, creates embedding of the new query calculated by concating them

        Args:
            queries (List[str])
            candidates (List[str])

        Returns:
            torch.Tensor
        """
       
        with torch.no_grad():
            sample_expanded_queries = [q + " " + c for q, c in zip(queries, candidates)]
            
            tokenized_new_q = self.tokenizer(
                sample_expanded_queries,padding="max_length",
                max_length=self.max_length ,
                truncation=True,
                return_tensors="pt",
                
            )

            tokenized_new_q = {k: v.to(self.device) for k, v in tokenized_new_q.items()}
            out = self.bert_model(**tokenized_new_q)
            emb = out[0]
            return emb
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_i = get_query_and_candidates_embeddings(
            [row["Q_name"]], [row["candidate_x"]], self.tokenizer, self.bert_model, self.max_length, self.device
        )
        x_j =  get_query_and_candidates_embeddings(
            [row["Q_name"]], [row["candidate_x"]], self.tokenizer, self.bert_model, self.max_length, self.device
        )   
        y_i = row["label_x"]
        y_j = row["label_y"]
        y = 1 if y_i > y_j else 0
        return x_i.squeeze(), x_j.squeeze(), y


class RankNetDatasetWithMap(Dataset):
    def __init__(self, pairs_df: pd.DataFrame, embedding_map: Dict[str, torch.Tensor]) -> None:

        self.query_x = (pairs_df['Q_name'] + " " + pairs_df['candidate_x']).tolist()
        self.query_y = (pairs_df['Q_name'] + " " + pairs_df['candidate_y']).tolist()
        # self.query_x = pairs_df['new_q_x'].tolist()
        # self.query_y = pairs_df['new_q_y'].tolist()
        self.label = (pairs_df['label_x'] < pairs_df['label_y']).astype(int).tolist()
        self.embeddings_map = embedding_map

    def __len__(self):
        return len(self.query_x)

    def __getitem__(self, idx):
        query_x = self.query_x[idx]
        query_y = self.query_y[idx]
        label = self.label[idx]

        x_embeddings = self.embeddings_map.get(query_x)
        if x_embeddings is None:
            raise Exception(f"'{query_x}' not in embeddings map")
        
        y_embeddings = self.embeddings_map.get(query_y)
        if y_embeddings is None:
            raise Exception(f"'{query_y}' not in embeddings map")

            


        return x_embeddings, y_embeddings, label

def get_fold_queries(fold_number: int, split: str = 'train', path: str = './data/queries_5_fold.csv', df: pd.DataFrame = None) -> List[str]:
    if df is None:
        if path:
            df = pd.read_csv(path)
        else:
            raise Exception("you must give one `path` or `df` for the fold csv file")
    
    final_df = df[df["fold"] == fold_number]
    if split:
        final_df = final_df[final_df["split"] == split]
    return final_df['Q_name'].tolist()

def create_pos_neg_pair_df(input_df: pd.DataFrame, max_sample_per_query: int = 200) -> pd.DataFrame:
    """given and dataframe with `Q_name` and `label` columns, create a new dataframe with each row including a pair of positive and negative labels

    Args:
        input_df (pd.DataFrame): input dataframe with `Q_name` and `label` columns. label should be binary (0, 1)
        max_sample_per_query (int, optional): how many candidates and sample per query should be used. 
                                              use this to limit the output size of your dataframe. Defaults to 200.

    Returns:
        pd.DataFrame: a dataframe including `Q_name`, `label_x`, `label_y`, `candidate_x` and `candidate_y`
    """

    df = input_df.groupby(by=["Q_name"]).head(max_sample_per_query)

    final_df = []

    for q_name in df["Q_name"].unique().tolist():
        df_idx: pd.DataFrame = df[df["Q_name"] == q_name]
        l1 = df_idx[df_idx["label"] == 1]
        l0 = df_idx[df_idx["label"] == 0]
        l0 = l0.sample(min(len(l1) * 2, len(l0)))
        df1 = pd.merge(l1, l0, on="Q_name")
        df0 = pd.merge(l0, l1, on="Q_name")
        final_df.append(pd.concat((df0, df1)))
    final_df = pd.concat(final_df)

    final_df = final_df.reset_index(drop=True)
    return final_df[['Q_name', 'candidate_x', 'label_x', 'candidate_y', 'label_y']]


if __name__ == '__main__':
    model_path = "./models/bert/Saved_model_epochs_5/"
    model = BertForMaskedLM.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    base_model = model.bert
    del model
    device = torch.device("cuda")
    base_model.to(device)

    df = pd.read_csv('./data/recall/expanded_and_labeled_queries_with_pubmed_recall_n100_200candidates.csv')
    fold_queries = get_fold_queries(fold_number=1, split= "train", path='./data/queries_5_fold.csv')
    fold_df = df[df["Q_name"].isin(fold_queries)]
    out_df = create_pos_neg_pair_df(fold_df, max_sample_per_query=20)
    dataset = RankNetDataset(out_df, tokenizer, bert=base_model, device=device)
    for d in dataset:
        print(d)
        exit(0)