import argparse
import pandas as pd
import scanpy as sc
import anndata as ad

def main(args):
    h5_file = f"{args.data_path}/filtered_feature_bc_matrix.h5"
    bed_file = f"{args.data_path}/atac_peaks.bed"
    
    peak = pd.read_csv(bed_file, sep='\t', names=['chr', 'start', 'end'], comment='#')
    sample_adata = sc.read_10x_h5(h5_file, gex_only=False)
    print('************* Raw Multiome data:')
    print(sample_adata)
    
    sample_adata.var["feature_types"] = sample_adata.var["feature_types"].map({"Gene Expression": "GEX", "Peaks": "ATAC"})
    sample_adata.var.index = sample_adata.var.index.str.replace(":", "-")
    sample_adata.var_names_make_unique()
    sample_adata.obs_names_make_unique()
    
    atac_idx = sample_adata.var["feature_types"] == "ATAC"
    sample_adata.var.loc[atac_idx, 'chr'] = peak['chr'].values
    sample_adata.var.loc[atac_idx, 'start'] = peak['start'].values
    sample_adata.var.loc[atac_idx, 'end'] = peak['end'].values
    
    sc.pp.filter_cells(sample_adata, min_genes=0)
    sc.pp.filter_genes(sample_adata, min_cells=0)
    # a gene/peak need to be expressed in 2% cells
    thres = int(sample_adata.shape[0] * 0.02)
    sample_adata = sample_adata[:, sample_adata.var['n_cells'] > thres]
    print('************* Filtered Multiome data:')
    print(sample_adata)
    
    sample_adata.layers["counts"] = sample_adata.X.copy()
    sample_adata.obs["atac_fragments"] = sample_adata.X.transpose()[sample_adata.var["feature_types"] == "ATAC"].transpose().sum(axis=1)
    sample_adata.write(f"{args.output_path}/multiome.h5ad")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess args.')
    
    parser.add_argument('--data_path', type=str, default='data/pbmc_3k/raw_data')
    parser.add_argument('--output_path', type=str, default='data/pbmc_3k/processed')
    args = parser.parse_args()

    main(args)