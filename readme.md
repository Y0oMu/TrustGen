# TrustEval-toolkit

*A modular and extensible toolkit for comprehensive trust evaluation of generative foundation models (GenFMs).*

![Overview](images/overview.jpg)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Trustworthiness Report](#trustworthiness-report)
- [Documentation](#documentation)
- [Support](#support)
- [License](#license)

---

## Overview

**TrustEval-toolkit** is a dynamic and comprehensive framework for evaluating the trustworthiness of Generative Foundation Models (GenFMs). It supports multi-dimensional evaluation across key trust areas, including safety, fairness, robustness, privacy, and truthfulness. The toolkit is designed to provide flexibility, extensibility, and actionable insights for researchers and developers aiming to build trustworthy AI systems.

---

## Features

- **Dynamic Dataset Generation**: Automatically generate datasets tailored for evaluation tasks.
- **Multi-Model Support**: Seamlessly evaluate large language models (LLMs), vision-language models (VLMs), and text-to-image (T2I) models.
- **Customizable Metrics**: Configure workflows with flexible metrics and evaluation methods.
- **Metadata-Driven Test Cases**: Efficiently design and execute test cases using metadata-driven pipelines.
- **Comprehensive Dimensions**: Evaluate models across areas such as safety, fairness, robustness, privacy, and truthfulness.
- **Efficient Inference**: Optimized inference pipelines for faster evaluations.
- **Detailed Reports**: Generate comprehensive and interactive evaluation reports.

---

## Installation

To install the **TrustEval-toolkit**, follow the steps below:

### 1. Clone the Repository

```bash
git clone https://github.com/nauyisu022/TrustEval-toolkit.git
cd TrustEval-toolkit
```

### 2. Set Up a Conda Environment

Create and activate a new environment with Python 3.10:

```bash
conda create -n trusteval_env python=3.10
conda activate trusteval_env
```

### 3. Install Dependencies

Install the package and its dependencies:

```bash
pip install .
```

---

## Usage

### Configure API Keys

Run the configuration script to set up your API keys:

```bash
python trusteval/src/configuration.py
```

![API Configuration](images/api_config.png)

### Quick Start
> *The following code examples should be run in a Jupyter Notebook.*

#### Example: Advanced AI Risk Evaluation

**Step 0:** Set your project base directory.
```python
import os
base_dir = os.getcwd() + '/advanced_ai_risk'
```

**Step 1:** Download metadata.
```python
from trusteval import download_metadata

download_metadata(
    section='advanced_ai_risk',
    output_path=base_dir
)
```

**Step 2:** Generate datasets dynamically.
```python
from trusteval.dimension.ai_risk import dynamic_dataset_generator

dynamic_dataset_generator(
    base_dir=base_dir,
)
```

**Step 3:** Apply contextual variations to the dataset.
```python
from trusteval import contextual_variator_cli

contextual_variator_cli(
    dataset_folder=base_dir
)
```

**Step 4:** Generate responses from your models.
```python
from trusteval import generate_responses

request_type = ['llm']  # Options: 'llm', 'vlm', 't2i'
async_list = ['your_async_model']
sync_list = ['your_sync_model']

await generate_responses(
    data_folder=base_dir,
    request_type=request_type,
    async_list=async_list,
    sync_list=sync_list,
)
```

**Step 5:** Evaluate the models.

- **Part 1: Judge the responses.**
    ```python
    from trusteval import judge_responses

    target_models = ['your_target_model1', 'your_target_model2']
    judge_type = 'llm'  # Options: 'llm', 'vlm', 't2i'
    judge_key = 'your_judge_key'
    async_judge_model = ['your_async_model']

    await judge_responses(
        data_folder=base_dir,
        async_judge_model=async_judge_model,
        target_models=target_models,
        judge_type=judge_type,
    )
    ```

- **Part 2: Generate evaluation metrics.**
    ```python
    from trusteval import lm_metric

    lm_metric(
        base_dir=base_dir,
        aspect='ai_risk',
        model_list=target_models,
    )
    ```

- **Part 3: Generate a detailed report.**
    ```python
    from trusteval import report_generator

    report_generator(
        base_dir=base_dir,
        aspect='ai_risk',
        model_list=target_models,
    )
    ```

Your `report.html` will be generated in the `base_dir` folder.

For other use cases, refer to the `examples` folder.

---

## Trustworthiness Report

A detailed trustworthiness evaluation report is generated for each dimension. The reports are presented as interactive web pages, which can be opened in a browser to explore the results. The evaluation report includes the following sections:

> *The data shown in the images below is simulated and does not reflect actual results.*

### **1. Test Model Results**
Displays the evaluation scores for each model, with a breakdown of average scores across evaluation dimensions.

![Test Model Results](images/test_model_results.png)

### **2. Model Performance Summary**
Summarizes the model's performance in the evaluated dimension using LLM-generated summaries, highlighting comparisons with other models.

![Model Performance Summary](images/model_performance_summary.png)

### **3. Error Case Study**
Presents error cases for the evaluated dimension, including input/output examples and detailed judgments.

![Error Case Study](images/error_case_study.png)

### **4. Leaderboard**
Shows the evaluation results for all models, along with visualized comparisons to previous versions (e.g., v1.0).

![Leaderboard](images/leaderboard.png)

---

## Documentation

For detailed documentation, including API references, tutorials, and best practices, please visit our official documentation site:

[TrustEval Documentation](https://trustgen.github.io/trustgen_docs/)

---

## Support

If you encounter any issues or have questions, you can:
1. Check our [documentation](https://trustgen.github.io/trustgen_docs/).
2. Open an issue in our [GitHub repository](https://github.com/nauyisu022/TrustEval-toolkit/issues).
3. Contact the maintainers.

---

## Video Tutorials

For visual demonstrations and step-by-step tutorials, visit our YouTube channel:

[TrustEval YouTube Channel](https://www.youtube.com/@TrustEval)

---

## License

This project is licensed under the [MIT License](LICENSE).