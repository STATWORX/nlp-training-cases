# NLP Training

Create a conda environment

- `conda create -n nlp-training`

- `conda activate nlp-training`

- `conda install --file requirements.txt -c conda-forge -c pytorch`

You can also try poetry

On Mac, install:

- `brew install cmake`

- `brew install rustup`

- `rustup-init`

- `source ~/.cargo/env`

- `rustc --version`

Then

- `pip install poetry`

- `poetry install`

- `poetry run python -m ipykernel install --user --name nlp-workshop`

On M1 Mac, you probably want to use conda
