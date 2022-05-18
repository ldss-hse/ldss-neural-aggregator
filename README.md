# Neural Turing Machine For Linguistic Decision-Making

This project is a part of the new direction in Decision Support Systems' evolution which is 
Neuro-Symbolic paradigm. To read more about Neuro-Symbolic AI, refer to the notable 
[paper](https://arxiv.org/abs/2012.05876)
by Artur d'Avila Garcez and Luis C. Lamb.

When applied to Decision Support Systems, several components of them could be subsymbolic, namely
defined as artificial neural networks. Some of these subsymbolic components could be trainable and require
huge generalization capabilities of performing intellectual tasks. Memory Augmented Neural Networks, and 
Neural Turing Machine in particular could be thought as a foundational models.

This repository contains all necessary code and scripts to perform training of Neural Turing Machines to solve 
numerous tasks, such as:

1. copy
2. associative recall
3. binary sum
4. binary average sum
5. MTA operator for linguistic decision-making.

Also, you can find a number of pre-trained [models](./trained_models) for most of the aforementioned tasks.

### Acknowledgements

1. The reported study was funded by RFBR, project number 19-37-90058.
2. This project has historical roots in 
   [Neural Turing Machine for TensorFlow contrib project](https://github.com/MarkPKCollier/NeuralTuringMachine), 
   although
   being tremendously improved and extended compared to the predecessor.

### Citing

To cite Neuro-Symbolic DSS advances produced by this project, you can cite the paper as follows:

```text
@inproceedings{demidovskij2021neural,
  title={Neural Multigranular 2-tuple Average Operator in Neural-Symbolic Decision Support Systems},
  author={Demidovskij, Alexander and Babkin, Eduard},
  booktitle={International Conference on Intelligent Information Technologies for Industry},
  pages={350--359},
  year={2021},
  organization={Springer}
}
```

### Contributing

#### Collecting code statistics about the project

1. Run:
   ```bash
   pygount --format=summary --folders-to-skip env,.git,venv,.idea,tpr_toolkit --suffix py,sh .
   ```
   
#### PEP8 style check

1. Run:

   ```bash
   pylint *.py tasks --ignore-paths=^tasks/operators/tpr_toolkit/.*$
   ```
