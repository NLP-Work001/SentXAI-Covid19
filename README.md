## **CI/CD: Sentiment Analysis for Covid-19 Tweets Using Explainable AI**

#### **Project Description:**

The dataset used in this project is open-sourced, [COVID-19 NLP Text Classification](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification/data). It consists of COVID-19 tweets that are annotated into five labels: negative, extremely negative, positive, extremely positive, and neutral. For our purpose, we consolidated "negative" and "extremely negative" into "negative," and "positive" and "extremely positive" into "positive." The labels of interest will be "negative," "neutral," and "positive."

The main objective is to compare different machine learning models and automate their experimentation workflow using Jenkins and GitHub Actions. Some of the tools used for experimentation will include MLflow and DVC. Furthermore, the best-performing model will be used in conjunction with LIME and SHAP values for interpretable and explainable artificial intelligence. Once model development and evaluation processes are completed, Django Rest Framework and Docker will be used to ensure that the model is easily containerized and deployed into the GitHub Container Registry. This will further ensure that the model can be tested through model serving using Docker images and be prepared for use in a web UI interface, although conducting the web UI for predictions is out of the scope of this project.

The main goal is to demonstrate the process of CI/CD automation in the field of machine learning while applying MLOps best practices.

#### **Project Substages:**

* **Stage 1 (notebooks/):**
    * Data cleaning and data loading: `notebook_v001`
    * Data analysis: `notebook_v002`
    * Model development and evaluation without thorough experimentation: `notebook_v003`

* **Stage 2: Model Development**

* **Stage 3: Model Deployment**
