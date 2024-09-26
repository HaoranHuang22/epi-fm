import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from model.datasets import GenomicDataModule
from model.model import FusionNetworkModule

file_config = {
    'seq_encoder': 'model/nucleotide-transformer-500m-human-ref',
    'cell_encoder': 'model/scFoundation/model/models/models.ckpt',
    'reference_genome': 'data/GRCh38_full_analysis_set_plus_decoy_hla.fa',
    'bed_exclude': 'data/exclude_regions.bed',
    'gene_table': 'data/OS_scRNA_gene_index.19264.tsv'
}

def main(args):
    h5ad_file = args.h5ad_file
    hidden_dim = args.hidden_dim
    fusion = args.fusion
    frozen_encoder = args.frozen_encoder
    is_regression = args.regression
    seq_length = args.seq_length
    task_setting = args.task_setting
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    lr = args.learning_rate

    seq_model_path = file_config['seq_encoder']
    rna_model_path = file_config['cell_encoder']
    reference_genome_file = file_config['reference_genome']
    bed_exclude = file_config['bed_exclude']
    gene_table = file_config['gene_table']
    if(is_regression):
        task_name = 'Reg'
    else:
        task_name = 'Class'
    
    genomic_dm = GenomicDataModule(
        h5ad_file,
        reference_genome_file=reference_genome_file,
        bed_exclude=bed_exclude,
        seq_length=seq_length,
        gene_table=gene_table,
        is_regression=is_regression,
        task_setting=task_setting,
        batch_size=batch_size,
        val_set=['chr2'],
        test_set=['chr3']
    )
    
    model = FusionNetworkModule(seq_model_path, rna_model_path, hidden_dim, is_regression, fusion=fusion, frozen_encoder=frozen_encoder, lr=lr)
    checkpoint_callback = ModelCheckpoint(
            monitor='valid_loss',
            dirpath=f"{h5ad_file}/checkpoints/",
            filename=f"{task_name}-best-checkpoint",
            save_top_k=1, 
            mode="min"
        )
    trainer = pl.Trainer(max_epochs=max_epochs, precision='16-mixed', callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=genomic_dm)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("This is the training script")
    parser.add_argument('--h5ad_file', type=str)
    parser.add_argument('--seq_length', type=int, default=1000)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--fusion', type=str, default='concat')
    parser.add_argument('--frozen_encoder', action='store_true')
    parser.add_argument('--regression', action='store_true')
    parser.add_argument('--task_setting', type=str, default='cross-region')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=str, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    
    args = parser.parse_args()
    main(args)