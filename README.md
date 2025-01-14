## **CI/CD: Sentiment Analysis for Covid-19 Tweets Using Explainable AI**

## **Project Description:**

The dataset used in this project is open-sourced, [COVID-19 NLP Text Classification](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification/data). It consists of COVID-19 tweets that are annotated into five labels: negative, extremely negative, positive, extremely positive, and neutral. For our purpose, we consolidated "negative" and "extremely negative" into "negative," and "positive" and "extremely positive" into "positive." The labels of interest will be "negative," "neutral," and "positive."

The main objective is to compare different machine learning models and automate their experimentation workflow using Jenkins and GitHub Actions. Some of the tools used for experimentation will include MLflow and DVC. Furthermore, the best-performing model will be used in conjunction with LIME and SHAP values for interpretable and explainable artificial intelligence. Once model development and evaluation processes are completed, Django Rest Framework and Docker will be used to ensure that the model is easily containerized and deployed into the GitHub Container Registry. This will further ensure that the model can be tested through model serving using Docker images and be prepared for use in a web UI interface, although conducting the web UI for predictions is out of the scope of this project.

The main goal is to demonstrate the process of CI/CD automation in the field of machine learning while applying MLOps best practices.

## **Project Setup**

**Prerequisites:**

* Git installed on your system (https://git-scm.com/downloads)
* Python 3.x installed on your system (https://www.python.org/downloads/)

**Steps:**

0. **Clone the project:**

   ```bash
   git clone[https://github.com/NLP-Work001/SentXAI-Covid19.git
   cd SentXAI-Covid19/
   ```

1. **Create data directory:**

   ```bash
   mkdir -p data/raw/covid-19-tweets
   ```

2. **Download COVID-19 tweets data:**

   Download the COVID-19 tweets dataset from [download-covid19](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification/data) and extract the contents into the `data/raw/covid-19-tweets` directory.

   The data should be separated into train and test CSV files. **Do not modify the file names.**

3. **Install requirements:**

   Set up a Python virtual environment to isolate project dependencies and activate it.  Activation commands might differ slightly based on your operating system:

   **Linux/macOS:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   **Windows:**

   ```bash
   python -m venv .venv
   venv\Scripts\activate
   ```

   Once activated, install the required packages using:

   ```bash
   pip install -r requirements.txt
   ```

