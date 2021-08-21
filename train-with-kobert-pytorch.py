#keras
#tensorflow
#torch
#pytorch_lightning
#transformers
#seqeval
#sentencepiece
"""
https://bhchoi.github.io/post/nlp/dev/bert_korean_spacing_04/
"""

import os
from typing import Callable, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import BertConfig, BertModel, AdamW
from torch.utils.data import Dataset, DataLoader
from seqeval.metrics import f1_score
import sys
from pathlib import Path
#sys.path.append(str(Path('./KoBERT').absolute()))
#from kobert.utils import get_tokenizer
sys.path.append(str(Path('./KoBERT-KorQuAD').absolute()))
from tokenization_kobert import KoBertTokenizer


config = lambda: None
setattr(config, 'task', 'korean_spacing_20210101')
setattr(config, 'log_path', 'logs')
setattr(config, 'bert_model', 'monologg/kobert')

setattr(config, 'train_data_path', 'TrainKoSpacing/data/1_완료_SCS93_A_연세대학교 법학연구원 글로벌비즈니스와 법센터.txt')
setattr(config, 'val_data_path', 'TrainKoSpacing/data/1_완료_SCS09_A_한국공연문화학회_2017_20210320_591.txt')
setattr(config, 'test_data_path', 'TrainKoSpacing/data/1_완료_SCS10_A_한국금융소비자학회_20210220_JKI.txt')
setattr(config, 'max_len', 128)
setattr(config, 'train_batch_size', 128)
setattr(config, 'eval_batch_size', 128)
setattr(config, 'dropout_rate', 0.1)
#setattr(config, 'gpus', 1)
setattr(config, 'gpus', None)   # do not use gpus
setattr(config, 'cpus', 12)  # for dataloader
setattr(config, 'distributed_backend', 'ddp')


class CorpusDataset(Dataset):
    def __init__(self, data_path: str, transform: Callable[[List, List], Tuple]):
        self.sentences = []
        self.slot_labels = ["UNK", "PAD", "B", "I"]
        self.transform = transform

        self._load_data(data_path)

    def _load_data(self, data_path: str):
        """data를 file에서 불러온다.

        Args:
            data_path: file 경로
        """
        with open(data_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            self.sentences = [line.split() for line in lines]

    def _get_tags(self, sentence: List[str]) -> List[str]:
        """문장에 대해 띄어쓰기 tagging을 한다.
        character 단위로 분리하여 BI tagging을 한다.

        Args:
            sentence: 문장

        Retrns:
            문장의 각 토큰에 대해 tagging한 결과 리턴
            ["B", "I"]
        """

        tags = []
        for word in sentence:
            for i in range(len(word)):
                if i == 0:
                    tags.append("B")
                else:
                    tags.append("I")
        return tags

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = "".join(self.sentences[idx])
        sentence = [s for s in sentence]
        tags = self._get_tags(self.sentences[idx])
        tags = [self.slot_labels.index(t) for t in tags]

        (
            input_ids,
            attention_mask,
            token_type_ids,
            slot_label_ids, 
        ) = self.transform(sentence, tags)

        return input_ids, attention_mask, token_type_ids, slot_label_ids

class Preprocessor:
    def __init__(self, max_len: int):
        self.tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
        self.max_len = max_len
        self.pad_token_id = 0

    def get_input_features(
        self, sentence: List[str], tags: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """문장과 띄어쓰기 tagging에 대해 feature로 변환한다.

        Args:
            sentence: 문장
            tags: 띄어쓰기 tagging

        Returns:
            feature를 리턴한다.
            input_ids, attention_mask, token_type_ids, slot_labels
        """

        input_tokens = []
        slot_label_ids = []
					
        # tokenize
        for word, tag in zip(sentence, tags):
            tokens = self.tokenizer.tokenize(word)

            if len(tokens) == 0:
                tokens = self.tokenizer.unk_token

            input_tokens.extend(tokens)

            for i in range(len(tokens)):
                if i == 0:
                    slot_label_ids.extend([tag])
                else:
                    slot_label_ids.extend([self.pad_token_id])

        # max_len보다 길이가 길면 뒤에 자르기
        if len(input_tokens) > self.max_len - 2:
            input_tokens = input_tokens[: self.max_len - 2]
            slot_label_ids = slot_label_ids[: self.max_len - 2]

        # cls, sep 추가
        input_tokens = (
            [self.tokenizer.cls_token] + input_tokens + [self.tokenizer.sep_token]
        )
        slot_label_ids = [self.pad_token_id] + slot_label_ids + [self.pad_token_id]

        # token을 id로 변환
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        # padding
        pad_len = self.max_len - len(input_tokens)
        input_ids = input_ids + ([self.tokenizer.pad_token_id] * pad_len)
        slot_label_ids = slot_label_ids + ([self.pad_token_id] * pad_len)
        attention_mask = attention_mask + ([0] * pad_len)
        token_type_ids = token_type_ids + ([0] * pad_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        slot_label_ids = torch.tensor(slot_label_ids, dtype=torch.long)

        return input_ids, attention_mask, token_type_ids, slot_label_ids


class SpacingBertModel(pl.LightningModule):
    def __init__(
        self,
        config,
        dataset: CorpusDataset,
    ):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.slot_labels_type = ["UNK", "PAD", "B", "I"]
        self.pad_token_id = 0

        self.bert_config = BertConfig.from_pretrained(
            self.config.bert_model, num_labels=len(self.slot_labels_type)
        )
        self.model = BertModel.from_pretrained(
            self.config.bert_model, config=self.bert_config
        )
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.linear = nn.Linear(
            self.bert_config.hidden_size, len(self.slot_labels_type)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        x = outputs[0]
        x = self.dropout(x)
        x = self.linear(x)

        return x

    def training_step(self, batch, batch_nb):

        input_ids, attention_mask, token_type_ids, slot_label_ids = batch

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        loss = self._calculate_loss(outputs, slot_label_ids)
        tensorboard_logs = {"train_loss": loss}

        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):

        input_ids, attention_mask, token_type_ids, slot_label_ids = batch

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        loss = self._calculate_loss(outputs, slot_label_ids)
        pred_slot_labels, gt_slot_labels = self._convert_ids_to_labels(
            outputs, slot_label_ids
        )

        val_f1 = self._f1_score(gt_slot_labels, pred_slot_labels)

        return {"val_loss": loss, "val_f1": val_f1}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_f1 = torch.stack([x["val_f1"] for x in outputs]).mean()

        tensorboard_log = {
            "val_loss": val_loss,
            "val_f1": val_f1,
        }

        return {"val_loss": val_loss, "progress_bar": tensorboard_log}

    def test_step(self, batch, batch_nb):

        input_ids, attention_mask, token_type_ids, slot_label_ids = batch

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pred_slot_labels, gt_slot_labels = self._convert_ids_to_labels(
            outputs, slot_label_ids
        )

        test_f1 = self._f1_score(gt_slot_labels, pred_slot_labels)

        test_step_outputs = {
            "test_f1": test_f1,
        }

        return test_step_outputs

    def test_epoch_end(self, outputs):
        test_f1 = torch.stack([x["test_f1"] for x in outputs]).mean()

        test_step_outputs = {
            "test_f1": test_f1,
        }

        return test_step_outputs

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.config.train_batch_size, num_workers=config.cpus)

    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.config.eval_batch_size, num_workers=config.cpus)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.config.eval_batch_size, num_workers=config.cpus)

    def _calculate_loss(self, outputs, labels):
        active_logits = outputs.view(-1, len(self.slot_labels_type))
        active_labels = labels.view(-1)
        loss = F.cross_entropy(active_logits, active_labels)

        return loss

    def _f1_score(self, gt_slot_labels, pred_slot_labels):
        return torch.tensor(
            f1_score(gt_slot_labels, pred_slot_labels), dtype=torch.float32
        )

    def _convert_ids_to_labels(self, outputs, slot_labels):
        _, y_hat = torch.max(outputs, dim=2)
        y_hat = y_hat.detach().cpu().numpy()
        slot_label_ids = slot_labels.detach().cpu().numpy()

        slot_label_map = {i: label for i, label in enumerate(self.slot_labels_type)}
        slot_gt_labels = [[] for _ in range(slot_label_ids.shape[0])]
        slot_pred_labels = [[] for _ in range(slot_label_ids.shape[0])]

        for i in range(slot_label_ids.shape[0]):
            for j in range(slot_label_ids.shape[1]):
                if slot_label_ids[i, j] != self.pad_token_id:
                    slot_gt_labels[i].append(slot_label_map[slot_label_ids[i][j]])
                    slot_pred_labels[i].append(slot_label_map[y_hat[i][j]])

        return slot_pred_labels, slot_gt_labels


if __name__ == "__main__":
    preprocessor = Preprocessor(config.max_len)

    dataset = {}
    dataset["train"] = CorpusDataset(
        config.train_data_path, preprocessor.get_input_features
    )
    dataset["val"] = CorpusDataset(
        config.val_data_path, preprocessor.get_input_features
    )
    dataset["test"] = CorpusDataset(
        config.test_data_path, preprocessor.get_input_features
    )

    bert_finetuner = SpacingBertModel(config, dataset)

    logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir=os.path.join(config.log_path, config.task), version=1, name=config.task
    )

    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        #filepath="checkpoints/"+ config.task + "/{epoch}_{val_loss:35f}",
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        #prefix="",
    )

    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=3,
        verbose=False,
        mode="min",
    )

    trainer = pl.Trainer(
        gpus=config.gpus,
        accelerator=config.distributed_backend,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=20,
        
    )
    # prevent "Early stopping conditioned on metric `val_loss` which is not available"
    trainer.fit(bert_finetuner) # Early stopping conditioned on metric `val_loss` which is not available. Pass in or modify your `EarlyStopping` callback to use any of the following: ``
    trainer.test()