import numpy as np
import torch
from transformers import BertModel, BertConfig, BertTokenizer

from numpy import ndarray
from hipo_rank import Document, Embeddings, SectionEmbedding, SentenceEmbeddings
from typing import List


class BertEmbedder:
    def __init__(self, bert_config_path: str, bert_model_path: str,
                 bert_tokenizer: str = "bert-base-cased",
                 bert_pretrained: str = None,
                 max_seq_len: int = 60,
                 cuda: bool = True):
        self.max_seq_len = max_seq_len
        self.cuda = cuda
        if bert_pretrained:
            self.bert_model = BertModel.from_pretrained(bert_pretrained)
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_pretrained)
        else:
            self.bert_model = self._load_bert(bert_config_path, bert_model_path)
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer)

    def _load_bert(self, bert_config_path: str, bert_model_path: str):
        bert_config = BertConfig.from_json_file(bert_config_path)
        model = BertModel(bert_config)
        if self.cuda:
            model_states = torch.load(bert_model_path)
        else:
            model_states = torch.load(bert_model_path, map_location='cpu')
        # fix model_states
        for k in list(model_states.keys()):
            if k.startswith("bert."):
                model_states[k[5:]] = model_states.pop(k)
            elif k.startswith("cls"):
                _ = model_states.pop(k)

        model.load_state_dict(model_states)
        if self.cuda:
            model.cuda()
        model.eval()
        return model

    def _get_sentences_embedding(self, sentences: List[str]) -> ndarray:
        # TODO: clean up batch approach
        input_ids = [self.bert_tokenizer.encode(s, add_special_tokens=True) for s in sentences]
        padded_len = min(max([len(x) for x in input_ids]), self.max_seq_len)
        num_inputs = len(input_ids)
        input_tensor = np.zeros((num_inputs, padded_len))
        for i,x in enumerate(input_ids):
            l = min(padded_len, len(x))
            input_tensor[i][:l] = x[:l]
        if self.cuda:
            input_tensor = torch.LongTensor(input_tensor).to('cuda')
        else:
            input_tensor = torch.LongTensor(input_tensor)
        batch_size = 20
        pooled_outputs = []
        for i in range(0, num_inputs, batch_size):
            input_batch = input_tensor[i:i+batch_size]
            # Original pacsum paper uses [CLS] next sentence prediction activations
            # this isn't optimal and should be changed for potentially better performance
            with torch.no_grad():
                pooled_output = self.bert_model(input_batch)[1] # shape = (x, 768)
            if self.cuda:
                pooled_output = pooled_output.cpu()
            else:
                pooled_output = pooled_output
            pooled_outputs.append(pooled_output)
        pooled_outputs = torch.cat(pooled_outputs).numpy()
        return pooled_outputs

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




