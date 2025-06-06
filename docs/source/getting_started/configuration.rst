Configuration
=============

1. Set Up a Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create and activate a new environment with Python 3.10:

.. code-block:: bash

    conda create -n trustgen_env python=3.10
    conda activate trustgen_env

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

Install the package and its dependencies:

.. code-block:: bash

    pip install .

🤖 Configure API Keys
---------------------

Run the configuration script to set up your API keys:

.. code-block:: bash

    python trusteval/src/configuration.py

.. image:: ../images/api_config.png
   :alt: API Configuration
