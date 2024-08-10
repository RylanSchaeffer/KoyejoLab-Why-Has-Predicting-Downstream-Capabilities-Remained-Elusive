# Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive?

This repository contains code and figures for our paper
[Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive?](https://arxiv.org/abs/2406.04391).

[![arXiv](https://img.shields.io/badge/arXiv-2406.04391-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2407.15211)


[**Setup**](#installation) | [**Usage**](#usage) | [**Contributing**](#contributing) | [**Citation**](#citation) | [**Contact**](#contact)


## Installation

1. (Optional) Update conda:

`conda update -n base -c defaults conda -y`

2. Create and activate the conda environment:

`conda env create --file environment.yml -y && conda activate elusive_env`

4. Update pip.

`pip install --upgrade pip`

5. Install some additional packages:

`pip install bitsandbytes sentencepiece`

6. (Optional) To run evals, initialize EleutherAI's `lm-evaluation-harness`:

`git submodule update --init --recursive`

Change into the directory and install `lm-evaluation-harness`:

`cd submodules/lm-evaluation-harness && pip install -e . && cd ../..`

7. Login to `wandb`:

`wandb login`

## Data

Data will be provided once the paper is accepted and published. For early access, please contact the 
authors [below](#contact).

## Usage


## Contributing

Contributions are welcome! Please format your code with [black](https://github.com/psf/black).

## Citation

To cite this work, please use:

```bibtex
@misc{schaeffer2024predictingdownstreamcapabilitiesfrontier,
      title={Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive?}, 
      author={Rylan Schaeffer and Hailey Schoelkopf and Brando Miranda and Gabriel Mukobi and Varun Madan and Adam Ibrahim and Herbie Bradley and Stella Biderman and Sanmi Koyejo},
      year={2024},
      eprint={2406.04391},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.04391}, 
}
```

Note: We created a new clean repository for the review process and the new commit history is not representative
of each individual's contributions.

## Contact

Questions? Comments? Interested in collaborating?
Open an issue or email rschaef@cs.stanford.edu and sanmi@cs.stanford.edu.
