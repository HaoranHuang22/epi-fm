import sys
import os
scFoundation_model_path = os.path.join(os.path.dirname(__file__), 'scFoundation', 'model')
sys.path.append(scFoundation_model_path)

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import get_peft_model, LoraConfig
from lora_pytorch import LoRA
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC, BinaryPrecision, BinaryRecall
from torchmetrics.regression import PearsonCorrCoef, SpearmanCorrCoef
from torchmetrics import MetricCollection
from load import *

class CellEncoder(nn.Module):
    def __init__(self, ckpt_path, frozen=True):
        super().__init__()
        model, model_config = load_model_frommmf(ckpt_path)
        self.model_config = model_config
        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder
        self.hidden_size = model_config['encoder']['hidden_dim']
  
        #encoder_layers = self.encoder.transformer_encoder
        #for layer in encoder_layers:
            #layer.self_attn = LoRA.from_module(layer.self_attn, rank=8)
        
        if frozen:
            for _, p in self.token_emb.named_parameters():
                p.requires_grad = False
            for _, p in self.pos_emb.named_parameters():
                p.requires_grad = False
            print('model position embedding and token embedding are frozen')
            
            for na, param in self.encoder.named_parameters():
                param.requires_grad = False
            print('cell encoder is frozen')
            
    def forward(self, x):
        value_labels = x > 0
        x, x_padding = gatherData(x, value_labels, self.model_config['pad_token_id'])
        data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels, self.model_config['pad_token_id'])
        
        x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
        position_emb = self.pos_emb(position_gene_ids)
        x += position_emb
        
        embeddings = self.encoder(x, x_padding)
        h = torch.mean(embeddings[:, :-2, :], dim=1)
        return h
        
class SequenceEncoder(nn.Module):
    def __init__(self, ckpt_path, frozen=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        self.encoder = AutoModelForMaskedLM.from_pretrained(ckpt_path).cuda()
        self.hidden_size = self.encoder.config.hidden_size
        self.device = self.encoder.device
        
        if frozen:
            for _, p in self.encoder.named_parameters():
                p.requires_grad = False
            print('sequence encoder is frozen')
        
    def forward(self, seqs):
        tokens_ids = self.tokenizer.batch_encode_plus(seqs, return_tensors="pt", padding="max_length", max_length = self.tokenizer.model_max_length)["input_ids"]
        attention_mask = tokens_ids != self.tokenizer.pad_token_id
        tokens_ids, attention_mask = tokens_ids.to(self.device), attention_mask.to(self.device)

        torch_outs = self.encoder(
            tokens_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True
        )
        embeddings = torch_outs['hidden_states'][-1]
        attention_mask = torch.unsqueeze(attention_mask, dim=-1)
        h = torch.sum(attention_mask*embeddings, axis=-2)/torch.sum(attention_mask, axis=1)
        return h
    
class NeuralTensorNetwork(nn.Module):
    def __init__(self, input_dim, tensor_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(input_dim, input_dim, tensor_dim))
        self.V = nn.Linear(2 * input_dim, tensor_dim)
        self.b = nn.Parameter(torch.randn(tensor_dim))
        self.out_layer = nn.Sequential(
            nn.Tanh(),
            nn.Linear(tensor_dim, 1)
        )
    
    def forward(self, e_1, e_2):
        t_1 = torch.einsum('bi,ijk,bj->bk', e_1, self.W, e_2)
        t_2 = torch.cat([e_1, e_2], dim=1)
        output = t_1 + self.V(t_2) + self.b
        output = self.out_layer(output)
        return output
    
class MLPAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1)
        )
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//4, 1)
        )
    
    def forward(self, e_1, e_2):
        e_1_attn = self.mlp_attn(e_1)
        e_2_attn = self.mlp_attn(e_2)
        attn_weights = torch.cat([e_1_attn, e_2_attn], dim=1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        e_1_weighted = e_1 * attn_weights[:, 0].unsqueeze(1)
        e_2_weighted = e_2 * attn_weights[:, 1].unsqueeze(1)
        attn_output = e_1_weighted + e_2_weighted
        
        output = self.out_layer(attn_output)
        return output
        
class FusionNetwork(nn.Module):
    def __init__(self, seq_model_path, rna_model_path, hidden_dim, is_regression, fusion='concat', frozen_encoder=True, tensor_dim=32):
        super().__init__()
        self.cell_encoder = CellEncoder(rna_model_path, frozen_encoder)
        self.seq_encoder = SequenceEncoder(seq_model_path, frozen_encoder)
        
        self.cell_down_proj = nn.Sequential(
            nn.Linear(self.cell_encoder.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.seq_down_proj = nn.Sequential(
            nn.Linear(self.seq_encoder.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.is_regression = is_regression

        if fusion == 'concat':
            self.fusion_layer = nn.Sequential(
                nn.Linear(2*hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim//4),
                nn.ReLU(),
                nn.Linear(hidden_dim//4, 1)
            )
        elif fusion == 'ntn':
            if tensor_dim is None:
                tensor_dim = hidden_dim
            self.fusion_layer = NeuralTensorNetwork(hidden_dim, tensor_dim)
        elif fusion == 'mlp_attn':
            self.fusion_layer = MLPAttention(hidden_dim)
        self.fusion = fusion    
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq, rna):
        seq_embed = self.seq_encoder(seq)
        seq_embed = self.seq_down_proj(seq_embed)
        
        cell_embed = self.cell_encoder(rna)
        cell_embed = self.cell_down_proj(cell_embed)
        
        if self.fusion == 'concat':
            embeds = torch.cat([seq_embed, cell_embed], dim=1)
            output = self.fusion_layer(embeds)
        else:
            output = self.fusion_layer(seq_embed, cell_embed)
        
        if not self.is_regression:
            output = self.sigmoid(output)
        return output
    
class FusionNetworkFromPretrained(FusionNetwork):
    def __init__(self, seq_model_path, rna_model_path, hidden_dim, is_regression, fusion='concat', frozen_encoder=True, tensor_dim=32):
        super().__init__(seq_model_path, rna_model_path, hidden_dim, is_regression, fusion=fusion, frozen_encoder=frozen_encoder, tensor_dim=tensor_dim)
    
    def forward(self, seq_embed, cell_embed):
        seq_embed = self.seq_down_proj(seq_embed)
        cell_embed = self.seq_down_proj(cell_embed)
        if self.fusion == 'concat':
            embeds = torch.cat([seq_embed, cell_embed], dim=1)
            output = self.fusion_layer(embeds)
        else:
            output = self.fusion_layer(seq_embed, cell_embed)
        
        if not self.is_regression:
            output = self.sigmoid(output)
        return output

class FusionNetworkModule(pl.LightningModule):
    def __init__(self, seq_model_path, rna_model_path, hidden_dim, is_regression, fusion='concat', frozen_encoder=True, pretrained=True, lr=1e-5):
        super().__init__()
        self.lr = lr
        self.is_regression = is_regression
        
        if pretrained:    
            self.model = FusionNetworkFromPretrained(seq_model_path, rna_model_path, hidden_dim, is_regression, fusion=fusion, frozen_encoder=frozen_encoder)
        else:
            self.model = FusionNetwork(seq_model_path, rna_model_path, hidden_dim, is_regression, fusion=fusion, frozen_encoder=frozen_encoder)
        if(is_regression):
            metrics = MetricCollection([PearsonCorrCoef(), SpearmanCorrCoef()])
        else:
            metrics = MetricCollection([BinaryAccuracy(), BinaryF1Score(), BinaryAUROC(), BinaryPrecision(), BinaryRecall()])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='valid_')
    
    def forward(self, seq, rna):
        return self.model(seq, rna)
    
    def process_batch(self, batch):
        seq, rna, target = batch
        pred = self(seq, rna)
        target = target.unsqueeze(1)
        if(self.is_regression):
            loss = torch.nn.MSELoss(reduction='mean')
        else:
            loss = torch.nn.BCELoss(reduction='mean')
        loss = loss(pred, target)
        return loss, seq, rna, target, pred

    def training_step(self, batch, batch_idx):
        loss, seq, rna, target, pred = self.process_batch(batch)
        
        pred_flat = pred.clone().view(-1)
        target_flat = target.clone().view(-1)
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict(self.train_metrics(pred_flat, target_flat), sync_dist=True, on_epoch=True, batch_size=rna.shape[0])
        return loss
    
    def on_train_epoch_end(self):
        self.train_metrics.reset()
        print('\n')
    
    def validation_step(self, batch, batch_idx):
        loss, seq, rna, target, pred = self.process_batch(batch)
        
        pred_flat = pred.clone().view(-1)
        target_flat = target.clone().view(-1)
        
        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict(self.valid_metrics(pred_flat, target_flat), sync_dist=True, on_epoch=True, batch_size=rna.shape[0])
        
    def on_validation_epoch_end(self):
        self.valid_metrics.reset()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)