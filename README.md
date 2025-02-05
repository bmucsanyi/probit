# probit

Probit and sigmoid inference for (approximate) exactness.

### Packages

The ImageNet-C and CIFAR-10C perturbations use [Wand](https://docs.wand-py.org/en/latest/index.html), a Python binding of [ImageMagick](https://imagemagick.org/index.php). Follow [these instructions](https://docs.wand-py.org/en/latest/guide/install.html) to install ImageMagick. Wand is installed below.

Create a virtual environment for `probit` by running `python -m venv` (or `uv venv`) in the root folder.
Activate the virtual environment with `source .venv/bin/activate` and run one of the following commands based on your use case:
- Work with the existing code: `python -m pip install .` (or `uv pip install .`)
- Extend the code base: `python -m pip install -e '.[dev]'` (or `uv pip install -e '.[dev]'`)

## Contributing

Contributions are very welcome. Before contributing, please make sure to run `pre-commit install`. Feel free to open a pull request with new methods or fixes.
