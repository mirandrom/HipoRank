import numpy as np
import torch
from transformers import BertModel, BertConfig, BertTokenizer

from numpy import ndarray
from hipo_rank import Document, Embeddings, SectionEmbedding, SentenceEmbeddings
from typing import List


class BertEmbedder:
    def __init__(self, bert_config_path: str, bert_model_path: str,
                 bert_tokenizer: str = "bert-base-cased",
                 bert_pretrained: str = None):
        if bert_pretrained:
            self.bert_model = BertModel.from_pretrained(bert_pretrained)
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_pretrained)
        else:
            self.bert_model = self._load_bert(bert_config_path, bert_model_path)
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer)

    def _load_bert(self, bert_config_path: str, bert_model_path: str):
        bert_config = BertConfig.from_json_file(bert_config_path)
        model = BertModel(bert_config)
        model_states = torch.load(bert_model_path, map_location='cpu')
        # fix model_states
        for k in list(model_states.keys()):
            if k.startswith("bert."):
                model_states[k[5:]] = model_states.pop(k)
            elif k.startswith("cls"):
                _ = model_states.pop(k)

        model.load_state_dict(model_states)
        model.eval()
        return model

    def _get_sentences_embedding(self, sentences: List[str]) -> ndarray:
        # TODO: clean up batch approach
        input_ids = [self.bert_tokenizer.encode(s, add_special_tokens=True) for s in sentences]
        padded_len = min(max([len(x) for x in input_ids]), 512)
        batch_size = len(input_ids)
        input_tensor = np.zeros((batch_size, padded_len))
        for i,x in enumerate(input_ids):
            input_tensor[i][:len(x)] = x
        input_tensor = torch.LongTensor(input_tensor)
        # Original pacsum paper uses [CLS] next sentence prediction activations
        # this isn't optimal and should be changed for potentially better performance
        with torch.no_grad():
             pooled_output = self.bert_model(input_tensor)[1].numpy() # shape = (x, 768)
        return pooled_output

    def get_embeddings(self, doc: Document) -> Embeddings:
        sentence_embeddings = []
        for section in doc.sections:
            id = section.id
            sentences = section.sentences
            se = self._get_sentences_embedding(sentences)
            sentence_embeddings += [SentenceEmbeddings(id=id, embeddings=se)]
        section_embeddings = [SectionEmbedding(id=se.id, embedding=np.mean(se.embeddings, axis=0))
                              for se in sentence_embeddings]

        return Embeddings(sentence=sentence_embeddings, section=section_embeddings)




