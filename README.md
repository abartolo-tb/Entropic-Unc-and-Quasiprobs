# Entropic-Unc-and-Quasiprobs
Numerical simulation code for the paper: Reconciling two notions of quantum operator disagreement: Entropic  uncertainty relations and information scrambling, united through  quasiprobabilities [1806.04147]

The Python file simulationTools.py contains a collection of tools for simulating and visualizing the behavior of spin chains. This file is used by the other Python scripts and Jupyter notebooks.

examples.ipynb is a convenient Jupyter notebook interface for using simulationTools.py. The Jupyter notebook starts with a guided example demonstrating the use of simulationTools.py for simulating the evolution of a spin chain and visualizing the various quantities of interest. The notebook also contains multiple examples with pre-specified parameter choices that produce interesting behavior. It should be cautioned that examples involving fine-grained projectors can take several hours to run, depending on the type of computer and number of workers.

create_weak_data.py and create_strong_data.py generate the datasets used in the paper. These scripts simulate the behavior of a Power-law Quantum Ising Model in the weak-coupling regime with coarse-grained projectors and the strong-coupling regime with fine-grained projectors, respectively. The create_strong_data.py script will take up to a week to run unless shared across multiple machines.

The folders weak_data/ and strong_data/ contain the datasets produced by create_weak_data.py and create_strong_data.py, respectively. Due to filesize limits on GitHub, rhoF.npy and rhoF.npy for create_strong_data.py have been omitted.

The Jupyter notebooks visualize-weak.ipynb and visualize-strong.ipynb visualize the datasets in weak_data/ and strong_data/, respectively. The plots generated in these notebooks are the same as the ones used in the paper.
