from pathlib import Path
import json
from hipo_rank.summarizers.lead import LeadSummarizer
from hipo_rank.summarizers.oracle import OracleSummarizer
from hipo_rank.dataset_iterators.pubmed import PubmedDataset
from hipo_rank.evaluators.rouge import evaluate_rouge
import time
from tqdm import tqdm

"""
Lead
"""
DEBUG = False

RESULTS_PATH = Path(f"results/exp5")
DATASETS = [
    ("pubmed_val", PubmedDataset, {"file_path": "data/pubmed-release/val.txt"}),
    ("arxiv_val", PubmedDataset, {"file_path": "data/arxiv-release/val.txt"}),
]
NUM_WORDS = {
    'pubmed_val': 200,
    'arxiv_val': 220
}
SUMMARIZERS = [
    ('oracle', OracleSummarizer, {}),
    ('lead', LeadSummarizer, {}),

]

for (dataset_id, dataset, dataset_args) in DATASETS:
    DataSet = dataset(**dataset_args)
    docs = list(DataSet)
    if DEBUG:
        docs = docs[:5]
    for (summarizer_id, summarizer, summarizer_args) in SUMMARIZERS:
        summarizer_args.update(dict(num_words=NUM_WORDS[dataset_id]))
        Summarizer = summarizer(**summarizer_args)
        experiment_path = RESULTS_PATH / f"{dataset_id}_{summarizer_id}"
        try:
            experiment_path.mkdir(parents=True)
            results = []
            references = []
            summaries = []
            for doc in tqdm(docs):
                summary = Summarizer.get_summary(doc)
                results.append({
                    "num_sects": len(doc.sections),
                    "num_sents": sum([len(s.sentences) for s in doc.sections]),
                    "summary": summary,

                })
                summaries.append([s[0] for s in summary])
                references.append([doc.reference])
            rouge_result = evaluate_rouge(summaries, references)
            (experiment_path / "rouge_results.json").write_text(
                json.dumps(rouge_result, indent=2))
            (experiment_path / "summaries.json").write_text(
                json.dumps(results, indent=2))
        except FileExistsError:
            print(f"{experiment_path} already exists, skipping...")
            pass


