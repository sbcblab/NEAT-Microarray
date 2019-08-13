## Neuroevolution as a Tool for Microarray Gene Expression Pattern Identification in Cancer Research

Welcome to this repository. Here you will find all the code needed to run the N3O algorithm for microarray classification and gene selection.

### ABOUT

Microarrays are still one of the major techniques employed to study cancer biology. However, the identification of expression patterns from microarray datasets is still a significant challenge to overcome. In this work, a new approach using Neuroevolution, a machine learning field that combines neural networks and evolutionary computation, provides aid in this challenge by simultaneously classifying microarray data and selecting the subset of more relevant genes. The main algorithm, FS-NEAT, was adapted by the addition of new structural operators designed for this high dimensional data. In addition, a rigorous filtering and preprocessing protocol was employed to select quality microarray datasets for the proposed method, selecting 13 datasets from three different cancer types. The results show that Neuroevolution was able to successfully classify microarray samples when compared with other methods in the literature, while also finding subsets of genes that can be generalized for other algorithms and carry relevant biological information. This approach detected 177 genes, and 82 were validated as already being associated to their respective cancer types and 44 were associated to other types of cancer, becoming potential targets to be explored as cancer biomarkers. Five long non-coding RNAs were also detected, from which four don’t have described functions yet. The expression patterns found are intrinsically related to extracellular matrix, exosomes and cell proliferation. The results obtained in this work could aid in unraveling the molecular mechanisms underlying the tumoral process and describe new potential targets to be explored in future works.

### CONTACT INFORMATION

Neuroevolution as a Tool for Microarray Gene Expression Pattern Identification in Cancer Research

Bruno Iochins Grisci, Bruno César Feltes, Márcio Dorn

Institute of Informatics, Federal University of Rio Grande do Sul, Porto Alegre, RS, Brazil

E-mail: bigrisci@inf.ufrgs.br and mdorn@inf.ufrgs.br

http://sbcb.inf.ufrgs.br

### CITATION

If you use N3O in a scientific publication, we would appreciate citations to the following paper:

Grisci, Bruno Iochins, Bruno César Feltes, and Marcio Dorn. "Neuroevolution as a Tool for Microarray Gene Expression Pattern Identification in Cancer Research." Journal of Biomedical Informatics (2018).
https://doi.org/10.1016/j.jbi.2018.11.013

Bibtex entry:

```
@article{grisci2019neuroevolution,
  title={Neuroevolution as a tool for microarray gene expression pattern identification in cancer research},
  author={Grisci, Bruno Iochins and Feltes, Bruno C{\'e}sar and Dorn, Marcio},
  journal={Journal of biomedical informatics},
  volume={89},
  pages={122--133},
  year={2019},
  publisher={Elsevier}
}
```

### HOW TO USE

Download the neat.zip file and extract with the password "sbcbjbi2018" without the quotation marks. To run use the following command:

python NEAT.py save_dir number_generations number_cores k_cross_validation data_file label_file index_file

- save_dir: (String) directory path where the results should be saved
- number_generations: (Int) number of generations for the Genetic Algorithm
- number_cores: (Int) number of cores available for parallelezing
- k_cross_validation: (Int) specifies the number of folds for stratified cross-validation, set to 0 for no cross-validation
- data_file: (String) file path to the .gct file with the genes expression data
- label_file: (String) file path to the .cls file with the label of the samples
- index_file: (String) file path to .txt file with the indexes of the samples of each fold in cross-validation. Use None to create folds automatically

Example: 
```
python NEAT.py results 100 4 3 data/samples.gct data/samples.cls None
```

The final gene selection will be saved as tr_selection.gct

The final accuracy will be saved as accuracy.txt

The final neural network will be saved as reqnet.svg

The final confusion matrix will be saved as cv_cm.svg

### DEPENDENCIES

Run 
```
python check_dep.py 
```
to check if all needed libraries are installed.

Graphviz software is also required: https://www.graphviz.org/

The NEAT-Python’s is also required: https://neat-python.readthedocs.io

Please note that the code should run with Python 2.7.

### DATA FORMAT

Please use the .gct and .cls file formats to run this script.

GCT definition: http://software.broadinstitute.org/cancer/software/genepattern/file-formats-guide#GCT

CLS definition: http://software.broadinstitute.org/cancer/software/genepattern/file-formats-guide#CLS

Warning: Please note that for the use with N3O all classification problems must be binary (the input files must contain exactly 2 classes!).

### MICROARRAY DATA FOR TESTS

If you need data for tests or benchmarks, or just want microarray datasets for your own experiments, check CuMiDa (Curated Microarray Database) at: http://sbcb.inf.ufrgs.br/cumida

All data used in this paper is available through CuMiDa.

Warning: Please note that for the use with N3O all classification problems must be binary (the input files must contain exactly 2 classes!). You can adapt multiclass problems (like some of the datasets available at CuMiDa) by using One-vs-All classification: https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest
