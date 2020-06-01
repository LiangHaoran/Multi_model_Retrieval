import torchvision.models as models
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig


class ImageNet(nn.Module):
    """
    image
    """
    def __init__(self):
        super(ImageNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=51, out_channels=1, kernel_size=1)
        self.fc1 = nn.Linear(2053, 128)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        # permute
        # compass box feature
        x = self.conv1(x)
        x = x.reshape(-1, 2053)
        # compass dim
        x = self.fc1(x)
        x = self.tanh(x)
        return x


class TextNet(nn.Module):
    """
    text
    """
    def __init__(self,  code_length):
        super(TextNet, self).__init__()

        modelConfig = BertConfig.from_pretrained('/home/poac/code/Multi_modal_Retrieval/experiments/pretrained_models/bert-base-uncased-config.json')
        self.textExtractor = BertModel.from_pretrained('/home/poac/code/Multi_modal_Retrieval/experiments/pretrained_models/bert-base-uncased-pytorch_model.bin', config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size

        self.fc = nn.Linear(embedding_dim, code_length)
        self.tanh = torch.nn.Tanh()

    def forward(self, tokens, segments, input_masks):
        output=self.textExtractor(tokens, token_type_ids=segments, attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]  #output[0](batch size, sequence length, model hidden dimension)

        hash_features = self.fc(text_embeddings)
        hash_features = self.tanh(hash_features)
        return hash_features


class EmbNet(nn.Module):
    """
    Embedding
    """
    def __init__(self, opt):
        super(EmbNet, self).__init__()
        self.emb = nn.Embedding(opt.max_voca, 128, padding_idx=0)
        self.conv1_1 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1)
        self.conv1_2 = nn.Conv1d(in_channels=146, out_channels=1, kernel_size=1)
        self.tanh = torch.nn.Tanh()

    def forward(self, query, label):
        # query
        query = self.emb(query)
        query = self.conv1_1(query)
        # label
        label = self.emb(label)
        label = self.conv1_2(label)
        return self.tanh(query).reshape(-1, 128), self.tanh(label).reshape(-1, 128)

