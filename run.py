from hipo_rank.dataset_iterators.pubmed import PubmedDataset

from hipo_rank.embedders.w2v import W2VEmbedder
from hipo_rank.embedders.rand import RandEmbedder
from hipo_rank.embedders.bert import BertEmbedder

from hipo_rank.similarities.cos import CosSimilarity

from hipo_rank.directions.undirected import Undirected
from hipo_rank.directions.order import OrderBased
from hipo_rank.directions.edge import EdgeBased

from hipo_rank.scorers.add import AddScorer
from hipo_rank.scorers.multiply import MultiplyScorer

from hipo_rank.summarizers.default import DefaultSummarizer
from hipo_rank.evaluators.rouge import evaluate_rouge

from pathlib import Path
import pickle
import time
from tqdm import tqdm

DATASETS = [
    ("pubmed_val", PubmedDataset, {"file_path": "data/pubmed-release/val.txt"}),
    ("pubmed_val_no_sections", PubmedDataset,
     {"file_path": "data/pubmed-release/val.txt", "no_sections": True}
     ),
]
EMBEDDERS = [
    ("biomed_w2v", W2VEmbedder, {"bin_path": "models/wikipedia-pubmed-and-PMC-w2v.bin"}),
    ("rand_200", RandEmbedder, {"dim": 200}),
    ("biobert", BertEmbedder,
     {"bert_config_path": "models/biobert_v1.1_pubmed/bert_config.json",
      "bert_model_path": "models/biobert_v1.1_pubmed/pytorch_model.bin",
      "bert_tokenizer": "bert-base-cased"}
     ),
    ("bert", BertEmbedder,
     {"bert_config_path": "",
      "bert_model_path": "",
      "bert_tokenizer": "bert-base-cased",
      "bert_pretrained": "bert-base-cased"}
     ),
]
SIMILARITIES = [
    ("cos", CosSimilarity, {}),
]
DIRECTIONS = [
    ("undirected", Undirected, {}),
    ("order", OrderBased, {}),
    ("edge", EdgeBased, {}),
    ("backloaded_edge", EdgeBased, {"u": 0.8}),
    ("frontloaded_edge", EdgeBased, {"u": 1.2}),
]

SCORERS = [
    ("add", AddScorer, {}),
    ("multiply", MultiplyScorer, {}),
]
Summarizer = DefaultSummarizer()

experiment_time = int(time.time())
results_path = Path(f"results/{experiment_time}")
results_path.mkdir(parents=True, exist_ok=True)


for embedder_id, embedder, embedder_args in EMBEDDERS:
    Embedder = embedder(**embedder_args)
    print("loaded embedder ", embedder_id)
    for dataset_id, dataset, dataset_args in DATASETS:
        DataSet = dataset(**dataset_args)
        print("loaded dataset ", dataset_id)
        for similarity_id, similarity, similarity_args in SIMILARITIES:
            Similarity = similarity(**similarity_args)
            print("loaded similarity ", similarity_id)
            for direction_id, direction, direction_args in DIRECTIONS:
                Direction = direction(**direction_args)
                print("loaded direction ", direction_id)
                for scorer_id, scorer, scorer_args in SCORERS:
                    Scorer = scorer(**scorer_args)
                    print("loaded scorer ", scorer_id)
                    experiment = f"{dataset_id}-{embedder_id}-{similarity_id}-{direction_id}-{scorer_id}"
                    results = []
                    try:
                        for doc in tqdm(DataSet):
                            embeds = Embedder.get_embeddings(doc)
                            sims = Similarity.get_similarities(embeds)
                            scores = Scorer.get_scores(sims)
                            summary = Summarizer.get_summary(doc, scores)
                            results.append({"summary": summary, "reference": doc.reference})

                        summaries = [[s[0] for s in r["summary"]] for r in results]
                        references = [[r["reference"]] for r in results]
                        rouge_result = evaluate_rouge(summaries, references)
                        experiment_path = results_path / experiment
                        experiment_path.mkdir(parents=True, exist_ok=True)
                        (experiment_path / "rouge_results.pkl").write_bytes(pickle.dumps(rouge_result))
                        (experiment_path / "summaries.pkl").write_bytes(pickle.dumps(results))

                    except Exception as e:
                        print("[EXCEPTION] for ", experiment)
                        print(doc)
                        print(e)

