# Open Catalyst Project models

[![CircleCI](https://circleci.com/gh/Open-Catalyst-Project/ocp.svg?style=shield)](https://circleci.com/gh/Open-Catalyst-Project/ocp)

This fork is based on the ocp-models [codebase](https://github.com/Open-Catalyst-Project/ocp) for the [Open Catalyst Project](https://opencatalystproject.org/).

It provides implementations for the Allegro model ([code](https://github.com/mir-group/allegro.git), [paper](https://arxiv.org/abs/2204.05249)) for catalysis taking arbitrary chemical structures as input to predict energy / forces / positions.
Specifically this codebase extracts and adapts the Allegro implementation based on [Nequip](https://github.com/mir-group/nequip) to the ocp-models codebase.

## Installation

The easiest way to install prerequisites is via [conda](https://conda.io/docs/index.html).

After installing [conda](http://conda.pydata.org/), run the following commands
to create a new [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html)
named `ocp-allegro` and install dependencies.

### GPU installation

Instructions are for PyTorch 1.10.0, CUDA 11.3 specifically.


First, check that CUDA is in your `PATH` and `LD_LIBRARY_PATH`, e.g.
```bash
$ echo $PATH | tr ':' '\n' | grep cuda
/public/apps/cuda/11.3/bin

$ echo $LD_LIBRARY_PATH | tr ':' '\n' | grep cuda
/public/apps/cuda/11.3/lib64
```

The exact paths may differ on your system.

Then install the dependencies:
```bash
conda env create -f env.allegro.yml
```
Activate the conda environment with `conda activate ocp-models`.

Install this package with `pip install -e .`.

Finally, install the pre-commit hooks:
```bash
pre-commit install
```

### Allegro requirements

To use Allegro for OCP we first need to install the Allegro code requirements.

Currently adapting the Allegro model for our purposes is only possibly by slightly modifying the Nequip codebase.
Therefore one must install our fork of Nequip from source by executing the following commands:

```bash
git clone https://github.com/B-Czarnetzki/nequip.git
cd nequip
pip install .
```

Finally install Allegro itself.

```bash
git clone --depth 1 https://github.com/mir-group/allegro.git
cd allegro
pip install .
```


## Download data,

Consult [DATASET.md](https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md) for dataset download links and instructions.

## Train and evaluate models

For a detailed description of how to train and evaluate models using the ocp-models codebase consult [TRAIN.md](https://github.com/Open-Catalyst-Project/ocp/blob/main/TRAIN.md).

To run an Allegro model on the OC20 dataset point to one of the allegro.yaml files. For example:
```bash
python main.py --mode train --config-yml configs/s2ef/200k/allegro/allegro.yaml
```
These configs point to an Allegro specific config file.
```bash
config_path: 'allegro/allegro_forces_config.yaml'
```
To change allegro model hyperparameters like the number of layers or lmax they have to be changed in that config file.

## Logging

This code supports W&B and Tensorboard logging. The logger can be changed in the config file.


## Citation

If you use this codebase in your work, consider citing:

```bibtex
@article{ocp_dataset,
    author = {Chanussot*, Lowik and Das*, Abhishek and Goyal*, Siddharth and Lavril*, Thibaut and Shuaibi*, Muhammed and Riviere, Morgane and Tran, Kevin and Heras-Domingo, Javier and Ho, Caleb and Hu, Weihua and Palizhati, Aini and Sriram, Anuroop and Wood, Brandon and Yoon, Junwoong and Parikh, Devi and Zitnick, C. Lawrence and Ulissi, Zachary},
    title = {Open Catalyst 2020 (OC20) Dataset and Community Challenges},
    journal = {ACS Catalysis},
    year = {2021},
    doi = {10.1021/acscatal.0c04525},
}
```

## License

[MIT](https://github.com/Open-Catalyst-Project/ocp/blob/master/LICENSE.md)
