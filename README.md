CONTACT INFORMATION

Neuroevolution as a Tool for Microarray Gene Expression Pattern Identification in Cancer Research

Bruno Iochins Grisci, Bruno César Feltes, Márcio Dorn

Institute of Informatics, Federal University of Rio Grande do Sul, Porto Alegre, RS, Brazil

E-mail: mdorn@inf.ufrgs.br

http://sbcb.inf.ufrgs.br

CITATION

If you use NEAT-Microarray in a scientific publication, we would appreciate citations to the following paper:

Grisci, Bruno Iochins, Bruno César Feltes, and Marcio Dorn. "Neuroevolution as a Tool for Microarray Gene Expression Pattern Identification in Cancer Research." Journal of Biomedical Informatics (2018).
https://doi.org/10.1016/j.jbi.2018.11.013

Bibtex entry:

@article{grisci2018neuroevolution,
  title={Neuroevolution as a Tool for Microarray Gene Expression Pattern Identification in Cancer Research},
  author={Grisci, Bruno Iochins and Feltes, Bruno C{\'e}sar and Dorn, Marcio},
  journal={Journal of Biomedical Informatics},
  year={2018},
  publisher={Elsevier}
}

HOW TO USE

python NEAT.py save_dir number_generations number_cores k_cross_validation data_file label_file index_file

- save_dir: (String) directory path where the results should be saved
- number_generations: (Int) number of generations for the Genetic Algorithm
- number_cores: (Int) number of cores available for parallelezing
- k_cross_validation: (Int) specifies the number of folds for stratified cross-validation, set to 0 for no cross-validation
- data_file: (String) file path to the .gct file with the genes expression data
- label_file: (String) file path to the .cls file with the label of the samples
- index_file: (String) file path to .txt file with the indexes of the samples of each fold in cross-validation. Use None to create folds automatically

Example: python NEAT.py results 100 4 3 data/samples.gct data/samples.cls None


The final gene selection will be saved as tr_selection.gct

The final accuracy will be saved as accuracy.txt

The final neural network will be saved as reqnet.svg

The final confusion matrix will be saved as cv_cm.svg

DEPENDECIES

Run python check_dep.py to check if all needed libraries are installed.

DATA FORMAT

Please use the .gct and .cls file formats to run this script.

GCT definition: http://software.broadinstitute.org/cancer/software/genepattern/file-formats-guide#GCT

CLS definition: http://software.broadinstitute.org/cancer/software/genepattern/file-formats-guide#CLS
