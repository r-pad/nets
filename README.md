# nets

Pytorch networks used by r-pad.

Excludes pytorch-geometric-based networks, (which can be found in [r-pad/pyg_libs](https://github.com/r-pad/pyg_libs)).

List of networks implemented:
*

# Installation

*All of the following should be done inside your venv or conda environment.*

Strongly recommend using a conda env, because it makes installing flash attention MUCH easier.

## Prerequisites

### DiPTv3

1. Choose a CUDA version you'll want to use! Should be compatible with everything... CUDA 12.4 is good, but so is 11.8.

2. Install CUDA toolkit inside your conda env, so that nvcc is available. For instance:

    ```bash
    conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit
    ```

2. Choose a torch version (including GPU) and install it. See [here](https://pytorch.org/get-started/locally/) for instructions.
   For instance:

   ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    ```
3. Install an appropriate torch_scatter version. See [here](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
    For instance:

    ````bash
        pip install torch_scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
    ````

4. Install spconv.

    ```bash
        # spconv (SparseUNet)
        # refer https://github.com/traveller59/spconv
        pip install spconv-cu124  # choose version match your local cuda version
    ```

5. Install flash_attention.

    ```bash
        pip install flash-attn --no-build-isolation
    ```

## Option 1: No changes needed (simplest).

If you want to use the architecutres as-is (or their building blocks), you can just do the following:

````bash

export NETS_MODULE=diptv3
pip install 'git+https://github.com/r-pad/nets.git#egg=nets[$NETS_MODULE]@main'

````

You can replace `main` with a specific branch or tag.

## Option 2: Need to be able to modify the code.

If you want to be able to modify the code, but don't want to contribute back, you can do the following:

````bash

export NETS_MODULE=diptv3
cd $CODE_DIRECTORY
git clone https://github.com/r-pad/nets.git
cd nets
pip install -e '.[$NETS_MODULE]'

````

You can then modify the code in `$CODE_DIRECTORY/nets` and the changes will be reflected in your environment. And if you want to contribute back, you can create a branch and PR.

# Usage
