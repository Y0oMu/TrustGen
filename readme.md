# TrustEval-toolkit

## Installation

To install and use the **TrustEval-toolkit**, follow the steps below:

### 1. Clone the repository

```bash
git clone https://github.com/nauyisu022/TrustEval-toolkit.git
cd TrustEval-toolkit
```

### 2. Set up a Conda environment

Create and activate a new environment with Python 3.10:
```bash
conda create -n trusteval_env python=3.10
conda activate trusteval_env
```

### 3. Install dependencies

Install the package and its dependencies:
```bash
pip install .
```

## Usage

### Configure API Keys

Run the configuration script to set up your API keys:
```bash
python trusteval/src/configuration.py
```

![image](images/api_config.png)

### Quick Start

See our example notebooks in the `examples` directory for quick start guides:

- `fairness_llm.ipynb`: Evaluate LLM fairness
- `fairness_t2i.ipynb`: Evaluate text-to-image fairness 
- `fairness_vlm.ipynb`: Evaluate vision-language model fairness
- `privacy_llm.ipynb`: Evaluate LLM privacy
- `privacy_t2i.ipynb`: Evaluate text-to-image privacy
- `robustness_llm.ipynb`: Evaluate LLM robustness
- `robustness_t2i.ipynb`: Evaluate text-to-image robustness
- `safety_t2i.ipynb`: Evaluate text-to-image safety
- `truthfulness_llm.ipynb`: Evaluate LLM truthfulness

## Documentation

For detailed documentation, including API references, tutorials, and best practices, please visit our comprehensive documentation site:

[TrustEval Documentation](https://trustgen.github.io/trustgen_docs/)

## Support

If you encounter any issues or have questions, please:
1. Check our [documentation](https://trustgen.github.io/trustgen_docs/)
2. Open an issue in our GitHub repository 
3. Contact our maintainers

## Video Tutorials

For visual demonstrations and tutorials, check out our YouTube channel:
[TrustEval YouTube Channel](https://www.youtube.com/@TrustEval)
---

## License

This project is licensed under the [MIT License](LICENSE).