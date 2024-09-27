import pandas as pd
import pyranges as pr
import h5py
import torch
import random
import bisect
import scipy
import time
import re
import numpy as np
import torch.nn.functional as F
import scanpy as sc
import pytorch_lightning as pl
from Bio import SeqIO
from scipy import sparse
from tqdm import tqdm
from random import sample
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
import sys
import os
scFoundation_model_path = os.path.join(os.path.dirname(__file__), 'scFoundation', 'model')
sys.path.append(scFoundation_model_path)
from load import *

normal_chromosomes = ["chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8", "chr9", "chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20","chr21","chr22"]
unwanted_chars = "U|R|Y|K|M|S|W|B|D|H|V|N"

def main_gene_selection(X_df, gene_list):
    """
    Describe:
        rebuild the input adata to select target genes encode protein 
    Parameters:
        adata->`~anndata.AnnData` object: adata with var index_name by gene symbol
        gene_list->list: wanted target gene 
    Returns:
        adata_new->`~anndata.AnnData` object
        to_fill_columns->list: zero padding gene
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))), 
                              columns=to_fill_columns, 
                              index=X_df.index)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                        index=X_df.index, 
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns,var

class GenomicDataSet(Dataset):
    def __init__(self, h5ad_file, reference_genome_file, bed_exclude, chromosomes, seq_length, gene_table, 
                 seq_encoder=None, cell_encoder=None, cache_dir='./cache', use_precomputed=True, is_regression=False):
        """
        Initialize the dataset.
        :param seq_model: Pre-trained sequence embedding model (optional, used if `use_precomputed` is True).
        :param cell_model: Pre-trained cell embedding model (optional, used if `use_precomputed` is True).
        :param cache_dir: Directory to save and load cached embeddings.
        :param use_precomputed: Whether to use precomputed embeddings or not.
        """
        self.seq_length = seq_length
        self.gene_table = gene_table
        self.cache_dir = cache_dir
        self.use_precomputed = use_precomputed
        os.makedirs(cache_dir, exist_ok=True)
        
        bed_exclude_df = pd.read_csv(bed_exclude, sep="\t", header=None, usecols=[*range(0, 3)], names=["Chromosome", "Start", "End"])
        self.bed_exclude = pr.PyRanges(bed_exclude_df)
        
        reference_genome = self.load_reference(reference_genome_file, chromosomes)
        reference_genome = pr.PyRanges(reference_genome).subtract(self.bed_exclude)
        self.is_regression = is_regression
            
        data = sc.read_h5ad(h5ad_file)
        self.cell_names = data.obs.index.tolist()
        self.peaks = (data.var["feature_types"] == "ATAC") & (data.var.index.str.startswith(tuple([x+"-" for x in chromosomes])))
        self.peaks_count = sum(self.peaks)
        if is_regression:
            self.actual_peaks = data.layers["counts"][:,self.peaks]
        else:
            self.actual_peaks = data.X[:,self.peaks]
        self.peaks_names = self.peaks.index[self.peaks]
        self.total_samples = self.actual_peaks.count_nonzero()*2
        self.cells_starting_i = [-1]
        list_of_cells = []
        all_indices = np.arange(self.peaks_count)
        
        gexpr_feature = data[:, data.var['feature_types'] == 'GEX']
        self.genes = self.prepare_rna(gexpr_feature)
        
        for i, cell in tqdm(enumerate(self.actual_peaks), total=self.actual_peaks.shape[0]): # tierating through cells
            positive_indicies = cell.nonzero()[1]
            positive_count = len(positive_indicies)
            negative_indices = np.setdiff1d(all_indices, positive_indicies)
            
            if len(negative_indices) > positive_count:
                negative_indices = sample(list(set([x for x in range(0, self.peaks_count)])-set(positive_indicies)), positive_count)
                self.cells_starting_i.append(self.cells_starting_i[-1]+positive_count*2)
            else:
                negative_count = len(negative_indices)
                self.cells_starting_i.append(self.cells_starting_i[-1]+positive_count+negative_count)
                
            if(is_regression):
                cell_indices = list(positive_indicies)
                cell_indices.extend(negative_indices)
                cell[0,cell_indices] = np.log(cell[0,cell_indices].toarray()/data.obs.iloc[i]["atac_fragments"]*10000+1)+1
            else:
                cell[0,positive_indicies] = 1
                cell[0,negative_indices] = -1
            list_of_cells.append(cell)
      
        self.cells_starting_i.pop(0)
        self.actual_peaks = scipy.sparse.vstack(list_of_cells)
        
        self.seq_tokenizer = AutoTokenizer.from_pretrained(seq_encoder)
        if self.use_precomputed:
            self.seq_encoder = AutoModelForMaskedLM.from_pretrained(seq_encoder).cuda()
            self.seq_encoder.eval()
            self.device = self.seq_encoder.device
            
            self.cell_encoder, self.cell_encoder_config = load_model_frommmf(cell_encoder)
            self.cell_encoder.eval()
            
            self.precomputed_seq_embeddings = self.load_or_precompute_embeddings("seq")
            self.precomputed_cell_embeddings = self.load_or_precompute_embeddings("cell")
    
    def load_or_precompute_embeddings(self, embedding_type):
        """
        Load precomputed embeddings from disk, or precompute and save them if not already cached.
        :param embedding_type: 'seq' for sequence embeddings, 'cell' for cell embeddings.
        :return: List of precomputed embeddings.
        """
        cache_file = os.path.join(self.cache_dir, f"{embedding_type}_embeddings.pt")
        
        if os.path.exists(cache_file):
            print(f"Loading precomputed {embedding_type} embeddings from {cache_file}")
            embeddings = torch.load(cache_file)
        else:
            print(f"Precomputing {embedding_type} embeddings...")
            embeddings = {}
            
            if embedding_type == "seq":
                for i, peak in tqdm(enumerate(self.peaks_names.to_list()), total=len(self.peaks_names.to_list())):
                    chromosome, start, end = peak.split('-')
                    start = int(start)
                    end = int(end)
                    middle_of_peak = (start+end)//2
                    start_peak = middle_of_peak - self.seq_length//2
                    end_peak = middle_of_peak + self.seq_length//2
                    sequence = self.chr_seq[chromosome][start_peak:end_peak]
                    seq_embedding = self.precomputed_seq_embeddings(sequence)
                    embeddings[sequence] = seq_embedding.detach().cpu()
            
            elif embedding_type == "cell":
                for cell_id in tqdm(range(self.genes.shape[0])):
                    cell_name = self.genes.index[cell_id]
                    cell_embedding = self.precomputed_cell_embeddings(cell_id)
                    embeddings[cell_name] = cell_embedding.detach().cpu()
            
            torch.save(embeddings, cache_file)
            print(f"Saved {embedding_type} embeddings to {cache_file}")
        return embeddings
            
    def precomputed_seq_embeddings(self, sequence):
        tokens_ids = self.seq_tokenizer(sequence, return_tensors="pt", padding="max_length", max_length = self.seq_tokenizer.model_max_length)["input_ids"]
        attention_mask = tokens_ids != self.seq_tokenizer.pad_token_id
        tokens_ids, attention_mask = tokens_ids.to(self.device), attention_mask.to(self.device)
        
        with torch.no_grad():
            torch_outs = self.seq_encoder(
                tokens_ids,
                attention_mask=attention_mask,
                encoder_attention_mask=attention_mask,
                output_hidden_states=True
            )
        embeddings = torch_outs['hidden_states'][-1]
        attention_mask = torch.unsqueeze(attention_mask, dim=-1)
        seq_embeddings = torch.sum(attention_mask*embeddings, axis=-2)/torch.sum(attention_mask, axis=1)
        return seq_embeddings
    
    def precomputed_cell_embeddings(self, cell_id):
        tmpdata = (self.genes.iloc[cell_id,:]).tolist()
        totalcount = self.genes.iloc[cell_id,:].sum()
        pretrain_gene_x = torch.tensor(tmpdata+[4.0, np.log10(totalcount)]).unsqueeze(0).cuda()
        
        value_labels = pretrain_gene_x > 0
        x, x_padding = gatherData(pretrain_gene_x, value_labels, self.cell_encoder_config['pad_token_id'])
        data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels, self.cell_encoder_config['pad_token_id'])
        
        with torch.no_grad():
            x = self.cell_encoder.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
            position_emb = self.cell_encoder.pos_emb(position_gene_ids)
            x += position_emb
            
            embeddings = self.cell_encoder.encoder(x, x_padding)
            cell_embeddings = torch.mean(embeddings[:, :-2, :], dim=1)
        return cell_embeddings
    
    def __len__(self):
        return self.total_samples
    
    def get_cellid_peakid(self, id):
        cell_id = bisect.bisect_left(self.cells_starting_i, id)
        peak_id = self.actual_peaks[cell_id].nonzero()[1][id-(self.cells_starting_i[cell_id])]
        return cell_id, peak_id
    
    def __getitem__(self, idx):
        cell_id, peak_id = self.get_cellid_peakid(idx)
        peak = self.actual_peaks[cell_id, peak_id]
        
        if not(self.is_regression):
            if(peak < 0):
                peak = 0
        else:
            peak -= 1
        label = torch.tensor(peak).float()
        
        peak_name = self.peaks_names[peak_id]
        chromosome, start, end = peak_name.split('-')
        start = int(start)
        end = int(end)
        middle_of_peak = (start+end)//2
        start_peak = middle_of_peak - self.seq_length//2
        end_peak = middle_of_peak + self.seq_length//2
        sequence = self.chr_seq[chromosome][start_peak:end_peak]
        
        cell_name = self.cell_names[cell_id]
        
        if self.use_precomputed:
            seq_embedding = self.precomputed_seq_embeddings[sequence].squeeze(0)
            cell_embedding = self.precomputed_cell_embeddings[cell_name].squeeze(0)
            return seq_embedding, cell_embedding, label
        else:
            tokens_ids = self.seq_tokenizer(sequence, return_tensors="pt", padding="max_length", max_length = self.seq_tokenizer.model_max_length)["input_ids"]
            attention_mask = tokens_ids != self.seq_tokenizer.pad_token_id
            seq_inputs = {'input_ids': tokens_ids.squeeze(0),
                          'attention_mask': attention_mask.squeeze(0)}
            
            tmpdata = (self.genes.iloc[cell_id,:]).tolist()
            totalcount = self.genes.iloc[cell_id,:].sum()
            pretrain_gene_x = torch.tensor(tmpdata+[4.0, np.log10(totalcount)])
            value_labels = pretrain_gene_x > 0
            x, x_padding = gatherData(pretrain_gene_x, value_labels, self.model_config['pad_token_id'])
            data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
            position_gene_ids, _ = gatherData(data_gene_ids, value_labels, self.model_config['pad_token_id'])
            cell_inputs = {'gene_value': x,
                           'gene_ids': position_gene_ids}
            
            return seq_inputs, cell_inputs, label
    
    def prepare_rna(self, gexpr_feature):
        """
        return a gene value table
        """
        idx = gexpr_feature.obs_names.tolist()
        col = gexpr_feature.var.index.tolist()
        if sparse.issparse(gexpr_feature.X):
            gexpr_feature = gexpr_feature.X.toarray()
        else:
            gexpr_feature = gexpr_feature
        gexpr_feature = pd.DataFrame(gexpr_feature, index=idx, columns=col)
        
        adata = sc.AnnData(gexpr_feature)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        gexpr_feature = pd.DataFrame(adata.X,index=adata.obs_names,columns=adata.var_names)
        
        gene_list_df = pd.read_csv(self.gene_table, header=0, delimiter='\t')
        gene_list = list(gene_list_df['gene_name'])
        
        print('covert gene feature into 19264')
        gexpr_feature, to_fill_columns,var = main_gene_selection(gexpr_feature,gene_list)
        assert gexpr_feature.shape[1]>=19264
        
        return gexpr_feature
        
    def load_reference(self, reference_genome_file, chromosomes):
        self.chr_seq = dict()
        with open(reference_genome_file) as handle:
            seq_records = SeqIO.parse(handle, "fasta")
            for record in seq_records:
                if not(record.id in normal_chromosomes) or not(record.id in chromosomes): continue
                self.chr_seq[record.id] = str(record.seq)
            reference_genome = pd.DataFrame({"Chromosome": self.chr_seq.keys(), "Start": [1_100_000]*len(self.chr_seq.keys()), "End": [len(self.chr_seq[x])-1_100_000 for x in self.chr_seq.keys()]})
            return reference_genome
        
class GenomicDataModule(pl.LightningDataModule):
    def __init__(self, 
                 h5ad_file, 
                 reference_genome_file, 
                 bed_exclude, 
                 seq_length, 
                 gene_table,
                 seq_model=None,
                 cell_model=None,
                 use_precomputed=False,
                 is_regression=False,
                 task_setting="cross-region",
                 val_set=None,
                 test_set=None,
                 batch_size=16,
                 num_workers=4,
                 cache_dir='./cache'):
        super().__init__()
        self.h5ad_file = h5ad_file
        self.reference_genome_file = reference_genome_file
        self.bed_exclude = bed_exclude
        self.seq_length = seq_length
        self.gene_table = gene_table
        self.is_regression = is_regression
        self.task_setting = task_setting
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_model = seq_model
        self.cell_model = cell_model
        self.use_precomputed = use_precomputed
        self.cache_dir = cache_dir
        
        if self.task_setting == "cross-region":
            self.val_chr = val_set if val_set else ["chr2"]
            self.test_chr = test_set if test_set else ["chr3"]
        elif self.task_setting == "cross-cell":
            pass
        elif self.task_setting == "cross-both":
            pass
        else:
            raise ValueError(f"Unsupported task setting: {self.task_setting}")
        
    def setup(self, stage=None):
        if self.task_setting == "cross-region":
            train_chrs = [x for x in normal_chromosomes if x not in self.val_chr + self.test_chr]
            if (len(train_chrs) > 0):
                self.genomic_train = GenomicDataSet(
                    self.h5ad_file, 
                    self.reference_genome_file, 
                    self.bed_exclude, 
                    train_chrs, 
                    self.seq_length, 
                    self.gene_table, 
                    self.seq_model, 
                    self.cell_model, 
                    self.cache_dir, 
                    self.use_precomputed, 
                    self.is_regression
                    )
                
            if (len(self.val_chr) > 0):
                self.genomic_val = GenomicDataSet(
                    self.h5ad_file, 
                    self.reference_genome_file, 
                    self.bed_exclude, 
                    self.val_chr, 
                    self.seq_length, 
                    self.gene_table,
                    self.seq_model,
                    self.cell_model,
                    self.cache_dir,
                    self.use_precomputed, 
                    self.is_regression
                    )
                
            if (len(self.test_chr) > 0):
                self.genomic_test = GenomicDataSet(
                    self.h5ad_file, 
                    self.reference_genome_file, 
                    self.bed_exclude, 
                    self.test_chr, 
                    self.seq_length, 
                    self.gene_table,
                    self.seq_model,
                    self.cell_model,
                    self.cache_dir,
                    self.use_precomputed,
                    self.is_regression
                    )
    
    def train_dataloader(self):
        return DataLoader(self.genomic_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.genomic_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.genomic_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
if __name__ == "__main__":
    h5ad_file = "data/pbmc_3k/processed/multiome.h5ad"
    reference_genome_file = "data/GRCh38_full_analysis_set_plus_decoy_hla.fa"
    exclude_regions = "data/exclude_regions.bed"
    seq_length = 1000
    gene_table = "data/OS_scRNA_gene_index.19264.tsv"
    seq_encoder = "model/nucleotide-transformer-500m-human-ref"
    cell_encoder = "model/scFoundation/model/models/models.ckpt"
    cache_dir = './cache'
    use_precomputed = True
    is_regression = False
    task_setting = "cross-region"
    
    #genomic_data = GenomicDataSet(h5ad_file, reference_genome_file, exclude_regions, normal_chromosomes, seq_length, gene_table, seq_encoder, cell_encoder, cache_dir, use_precomputed)
    genomic_data_module = GenomicDataModule(h5ad_file, reference_genome_file, exclude_regions, seq_length, gene_table, seq_model=seq_encoder, cell_model=cell_encoder, use_precomputed=use_precomputed)
    genomic_data_module.setup()
    dl = genomic_data_module.val_dataloader()
    
    start_time = time.time() 
    for batch in dl:
        end_time = time.time()
        batch_time = end_time-start_time
        print(batch_time)
        start_time = time.time()
        
    
    
    