from hipo_rank.dataset_iterators.pubmed import PubmedDataset

from hipo_rank.embedders.bert import BertEmbedder
from hipo_rank.embedders.rand import RandEmbedder
from hipo_rank.embedders.sent_transformers import SentTransformersEmbedder

from hipo_rank.similarities.cos import CosSimilarity

from hipo_rank.directions.order import OrderBased
from hipo_rank.directions.edge import EdgeBased
from hipo_rank.directions.undirected import Undirected

from hipo_rank.scorers.add import AddScorer

from hipo_rank.summarizers.default import DefaultSummarizer
from hipo_rank.evaluators.rouge import evaluate_rouge

from pathlib import Path
import json
import time
from tqdm import tqdm

"""
hiporank with similarity thresholding (B=0.3)
"""

DEBUG = False

DATASETS = [
    ("pubmed_test", PubmedDataset,
     {"file_path": "data/pubmed-release/test.txt"}
     ),
    ("arxiv_test", PubmedDataset,
     {"file_path": "data/arxiv-release/test.txt"}
     ),
]
EMBEDDERS = [
    ("rand_20", RandEmbedder, {"dim": 20}),
    ("pacsum_bert", BertEmbedder,
     {"bert_config_path": "models/pacssum_models/bert_config.json",
      "bert_model_path": "models/pacssum_models/pytorch_model_finetuned.bin",
      "bert_tokenizer": "bert-base-uncased",
      }
    ),
    # ("st_bert_base", SentTransformersEmbedder,
    #      {"model": "bert-base-nli-mean-tokens"}
    #     ),
    # ("st_roberta_large", SentTransformersEmbedder,
    #      {"model": "roberta-large-nli-mean-tokens"}
    #     ),
]
SIMILARITIES = [
    ("cos", CosSimilarity, {"threshold": 0.3}),
]
DIRECTIONS = [
    ("order", OrderBased, {}),
    ("edge", EdgeBased, {}),
    ("undirected", Undirected, {}),
]

SCORERS = [
    ("add_f=0.0_b=1.0_s=1.0", AddScorer, {}),
]


SUMMARIZERS = [DefaultSummarizer(num_words=200),
               DefaultSummarizer(num_words=220)]

experiment_time = int(time.time())
results_path = Path(f"results/exp9")

for embedder_id, embedder, embedder_args in EMBEDDERS:
    Embedder = embedder(**embedder_args)
    for Summarizer, (dataset_id, dataset, dataset_args) in zip(SUMMARIZERS, DATASETS):
        DataSet = dataset(**dataset_args)
        docs = list(DataSet)
        if DEBUG:
            docs = docs[:5]
        print(f"embedding dataset {dataset_id} with {embedder_id}")
        embeds = [Embedder.get_embeddings(doc) for doc in tqdm(docs)]
        for similarity_id, similarity, similarity_args in SIMILARITIES:
            Similarity = similarity(**similarity_args)
            print(f"calculating similarities with {similarity_id}")
            sims = [Similarity.get_similarities(e) for e in embeds]
            for direction_id, direction, direction_args in DIRECTIONS:
                print(f"updating directions with {direction_id}")
                Direction = direction(**direction_args)
                sims = [Direction.update_directions(s) for s in sims]
                for scorer_id, scorer, scorer_args in SCORERS:
                    Scorer = scorer(**scorer_args)
                    experiment = f"{dataset_id}-{embedder_id}-{similarity_id}-{direction_id}-{scorer_id}"
                    experiment_path = results_path / experiment
                    try:
                        experiment_path.mkdir(parents=True)

                        print("running experiment: ", experiment)
                        results = []
                        references = []
                        summaries = []
                        for sim, doc in zip(sims, docs):
                            scores = Scorer.get_scores(sim)
                            summary = Summarizer.get_summary(doc, scores)
                            results.append({
                                "num_sects": len(doc.sections),
                                "num_sents": sum([len(s.sentences) for s in doc.sections]),
                                "summary": summary,

                            })
                            summaries.append([s[0] for s in summary])
                            references.append([doc.reference])
                        rouge_result = evaluate_rouge(summaries, references)
                        (experiment_path / "rouge_results.json").write_text(json.dumps(rouge_result, indent=2))
                        (experiment_path / "summaries.json").write_text(json.dumps(results, indent=2))
                    except FileExistsError:
                        print(f"{experiment} already exists, skipping...")
                        pass

