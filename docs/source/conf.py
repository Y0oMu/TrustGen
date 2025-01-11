# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sphinx_rtd_theme

project = 'TrustEval Docs'
copyright = '2025, TrustEval Teams'
author = 'TrustEval Teams'
release = '0.1.1'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # 自动生成文档
    'sphinx.ext.napoleon',  # 支持 Google 和 NumPy 风格的文档字符串
    'sphinx.ext.viewcode',  # 添加源代码链接
    'sphinx.ext.coverage',  # 检查文档覆盖率
    'sphinx.ext.intersphinx',  # 链接到其他项目的文档
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
