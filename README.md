# LLM Chatbot Project

This project is a chatbot application using LLM models such as GPT4All. The repository is structured to include different modules, GUI elements, model folder, and evaluation scripts.

## Directory Structure

- **chatmodules**: Contains Python modules for handling chatbot logic.
  - `gpt4all_chatbot.py`: Main script for the GPT4All chatbot logic.
  - `repeater.py`: A utility module that might be used for repeating or handling specific tasks related to chat functionalities.
  
- **gui**: Contains files related to the Graphical User Interface.
  - `main.ui`: UI configuration file for the project.
  - `requirements.txt`: Lists dependencies specific to the GUI.

- **models**: LLM files used for the chatbot should be placed here.

- **evaluation.ipynb**: Jupyter notebook for accuracy evaluating.
- **main.py**: Main entry point for the chatbot application.
- **README.md**: This file providing documentation for the project.
- **requirements.txt**: Python package dependencies for the entire project.
- **test_LLM.py**: Script for directly interacting with LLMs via Terminal.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8+ (Python 3.9 is recommended)
- Required packages listed in `requirements.txt`

You can install the dependencies using:
```bash
pip install -r requirements.txt