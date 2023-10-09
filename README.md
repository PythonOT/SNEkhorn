# SNEkhorn

Implementation of the paper [SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities](https://hal.science/hal-04103326) (NeurIPS 2023). SNEkhorn is a dimension reduction method based on optimal transport and symmetric entropic affinities. 

Feel free to ask if any question.

To intall the toobox run the following code in a shell:
```
pip install -e .
```

The main class can be imported as
```
from snekhorn import SNEkhorn
```

If you use this toolbox in your research and find it useful, please cite SNEkhorn using the following bibtex reference:

```
@inproceedings{snekhorn2023,
title={{SNE}khorn: Dimension Reduction with Symmetric Entropic Affinities},
author={Van Assel, Hugues and Vayer, Titouan and Flamary, R{\'e}mi and Courty, Nicolas},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)},
year={2023},
url={https://openreview.net/forum?id=y9U0IJ2uFr}
}
```

* Some simple demos are presented in the demo folder:
	- [example_affinities.py](/demo/example_affinities.py): illustrates the toolbox for calculating the symmetric entropic affinities.
	- [example_simple_snekhorn.py](/demo/example_simple_snekhorn.py): simple example that uses SNEkhorn.
    - [example_coil.py](/demo/example_coil.py): comparison of tSNE/tSNEkhorn on COIL dataset.

### Prerequisites

* Pytorch 
* Matplotlib
* Pillow

### Authors

* [Hugues Van Assel](https://huguesva.github.io/)
* [Titouan Vayer](https://tvayer.github.io/)
* [Rémi Flamary](https://remi.flamary.com/index.fr.html)
* [Nicolas Courty](https://people.irisa.fr/Nicolas.Courty/)
