# nets

Pytorch networks used by r-pad.

Excludes pytorch-geometric-based networks, (which can be found in [r-pad/pyg_libs](https://github.com/r-pad/pyg_libs)).

List of networks implemented:
*

# Installation

*All of the following should be done inside your venv or conda environment.*

## Option 1: No changes needed (simplest).

If you want to use the architecutres as-is (or their building blocks), you can just do the following:

````bash

pip install git+hhttps://github.com/r-pad/nets.git#egg=nets@main

````

You can replace `main` with a specific branch or tag.

## Option 2: Need to be able to modify the code.

If you want to be able to modify the code, but don't want to contribute back, you can do the following:

````bash

cd $CODE_DIRECTORY
git clone https://github.com/r-pad/nets.git
cd nets
pip install -e .

````

You can then modify the code in `$CODE_DIRECTORY/nets` and the changes will be reflected in your environment. And if you want to contribute back, you can create a branch and PR.

# Usage
