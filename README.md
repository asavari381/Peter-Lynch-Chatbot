# Peter-Lynch-Chatbot

This project implements a chatbot based on Peter Lynch's investment principles using **Transformer models**, including **GPT-2** and **Llama**. The chatbot leverages machine learning and natural language processing techniques to provide users with insights, strategies, and guidance inspired by Peter Lynch's iconic investment philosophies. 

Developed using **Python** and run on **Google Colab** / **Jupyter Notebook**, this project showcases the integration of advanced transformer models with financial knowledge for interactive user experiences.

---

## Features

- **Peter Lynch's Investment Principles**: Insights from Peter Lynch's PEG ratio analysis, growth metrics, and sector-specific strategies.
- **Transformer-Based Models**: GPT-2 and Llama models trained to respond with investment advice.
- **Interactive Chat Interface**: Engage in conversational-style interactions.
- **Custom Dataset Training**: Fine-tuned on datasets inspired by Lynch’s books, interviews, and investment practices.
- **Scalable and Portable**: Developed using Python for easy adaptation and deployment.

---

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Models](#models)
4. [Dataset](#dataset)
5. [How It Works](#how-it-works)
6. [Contributing](#contributing)
7. [License](#license)

---

## Installation

### Prerequisites
- Python 3.7 or higher
- Google Colab / Jupyter Notebook
- Required libraries: `transformers`, `torch`, `numpy`, `pandas`, `scikit-learn`

### Clone the Repository
```bash
git clone https://github.com/yourusername/Peter-Lynch-Chatbot.git
cd Peter-Lynch-Chatbot

Streamlit instructions for Macbook Users:
2. Install Streamlit
Use pip to install Streamlit:
pip3 install streamlit
3. Verify Installation
Confirm that Streamlit is installed by checking its version:
streamlit --version
4. Run the App (lynchapp.py)
Execute the app using the streamlit run command:
streamlit run lynchapp.py
5. Access the App
The app will open automatically in your default browser. If not, check the terminal output for a URL (e.g., http://localhost:8501) and open it manually.
6. Troubleshooting
Port Issues: If the default port is occupied, specify another port:
streamlit run app.py --server.port 8502
Environment Setup: Use a virtual environment for Streamlit to avoid conflicts:
python3 -m venv myenv
source myenv/bin/activate
pip install streamlit
