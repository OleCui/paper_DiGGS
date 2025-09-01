# paper_DiGGS
This is the PyTorch implementation for paper "Disentangled Hypergraph Neural Network for Drug Repositioning".

## Introduction


<img src='model.png'>

## Environment:
The codes of DiGGS are implemented and tested under the following development environment:
-  Python 3.8.19
-  cudatoolkit 11.5
-  pytorch 1.10.0
-  dgl 0.9.1
-  numpy 1.24.3
-  scikit-learn 1.3.0

## Datasets
We verify the effectiveness of our proposed method on three large-scale benchmarks, i.e., OpenBioLink, PrimeKG and BioKG.

These datasets can be downloaded from [OpenBioLink](https://drive.google.com/drive/folders/1K7yE0mgQOZsKCcDM-k_rYZFYmKmxWbrZ), [PrimeKG](https://drive.google.com/drive/folders/1jqJGuecCuvgi-0gNtQuPNv2HyXt-rFjO) and [BioKG](https://drive.google.com/drive/folders/1lVFo0o2dTV8FH6uhd73-x9RTOgLU-buL). Herein, we elaborate on the corresponding data files.
- <i>disease_to_idx.json</i>: The disease name to id mapping file.
- <i>drug_to_idx.json</i>: The drug name to id mapping file.
- <i>gene_to_idx.json</i>: The gene name to id mapping file.
- <i>phenotype_to_idx.json</i>: The phenotype name to id mapping file.
- <i>pathway_to_idx.json</i>: The pathway name to id mapping file.
- <i> disease-gene.csv </i>: The known disease-gene associations.
- <i> disease-phenotype.csv </i>: The known disease-phenotype associations.
- <i> disease-pathway.csv </i>: The known disease-pathway associations.
- <i> drug-gene.csv </i>: The known drug-gene associations.
- <i> drug-phenotype.csv </i>: The known drug-phenotype associations.
- <i> drug-pathway.csv </i>: The known drug-pathway associations.
- <i> drug-disease.csv </i>: The known drug-disease associations.

## Code Files:
The introduction of each <code> py </code> file is as follows:
- <i>dataloader.py</i>: Defines dataset classes and K-fold cross-validation data loaders for drug-disease association tasks.
- <i>hypergraph_data.py</i>: Constructs hypergraph datasets from biomedical knowledge graphs and implements negative sampling strategies for drug-disease pairs.
- <i>main.py</i>: Main entry point that orchestrates the training and evaluation pipeline with proper random seed initialization.
- <i>model_train.py</i>: Implements the training framework including metrics calculation, early stopping mechanism, and model evaluation procedures.
- <i>model.py</i>: Defines the decoupled hypergraph neural network architecture and drug-disease association predictor.
- <i>parse_args.py</i>: Configures command-line argument parsing for hyperparameter settings and experimental configurations.

## How to Run the Code:
Please firstly download the datasets and unzip the downloaded files. Next, create the <code>Datasets/</code> folder and move the unzipped datasets into this folder. The commands to train DiGGS on the OpenBioLink, PrimeKG or BioKG are as follows.

<ul>
<li>OpenBioLink<pre><code>python main.py --dataset = OpenBioLink</code></pre>
</li>
<li>PrimeKG<pre><code>python main.py --dataset = PrimeKG</code></pre>
</li>
<li>BioKG<pre><code>python main.py --dataset = BioKG</code></pre>
</li>
</ul>
</body></html>
