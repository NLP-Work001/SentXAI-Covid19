🚀 **Project Insights: Sentiment Analysis with Machine Learning Models and Explainable AI** 🚀

🔗 GitHub-link: https://github.com/baloyi015/sentiment-analysis-xai.git
🔗 Tools: Python & Jupyter-Notebook
My recent project delved into the effectiveness of various machine learning models and explainable AI for sentiment analysis, particularly focusing on tweet classification. Here’s a summary of the key learnings:

🔍 **Support Vector Machine (SVM)**
- **Skip-gram Word2Vec:** Although SVM accurately predicted negative sentiments, it misattributed neutral words (e.g., names, web links) to negative sentiments.
- **CBOW Word2Vec:** Similar misattributions were noted, with neutral words like "believe" and "social" incorrectly flagged as negative.
- **TF-IDF:** Demonstrated better performance by accurately attributing sentiments, though some inconsistencies remain.

🔍 **Logistic Regression**
- **Skip-gram Word2Vec:** Struggled with accuracy, often predicting incorrect sentiments.
- **CBOW Word2Vec:** More accurate in predicting negative sentiments but still faced misattribution issues.
- **TF-IDF:** Showcased the best performance, accurately predicting sentiments and correctly attributing words to their respective sentiments.

🌟 **Key Takeaway**
Logistic regression combined with TF-IDF emerged as the most reliable model, accurately predicting tweet sentiments and correctly attributing words. The integration of LIME (Local Interpretable Model-agnostic Explanations) further enhanced model transparency and interpretability, providing clear insights into decision-making processes.

🔗 **Why It Matters**
This project underscores the importance of model choice and data representation in sentiment analysis. By leveraging explainable AI tools, we can build models that are not only accurate but also transparent and trustworthy.

#MachineLearning #DataScience #SentimentAnalysis #NLP #AI #ModelInterpretability #ProjectInsights
