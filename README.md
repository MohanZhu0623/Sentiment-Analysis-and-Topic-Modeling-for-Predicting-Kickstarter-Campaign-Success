# Sentiment_Analysis

## Project Overview
This project aims to perform sentiment analysis and topic modeling on the Kickstarter dataset using various models. We used binary and ternary labeled sentiment analysis datasets to train and test various sentiment lexicons, traditional machine learning models, and transfer learning models. The best-performing models were then applied to a sample dataset of 20,000 rows. The project also includes topic modeling and exploratory visualization analysis. Additionally, different machine learning models were used to build predictive models for project success, comparing their performance and exploring the impact of various variables, including sentiment variables extracted from the text.

## File Structure

### Data Files
1. **binary_labelled_data.xlsx**
   - Description: Binary sentiment analysis data labeled for training models.

2. **python_labelled_data.xlsx**
   - Description: Ternary sentiment analysis data labeled for training models.

3. **kickstarter_data_sampled.xlsx**
   - Description: A sample dataset of 20,000 rows randomly extracted from the original dataset of approximately 170,000 rows.

4. **kickstarter_data_sampled_sentiment&WC.xlsx**
   - Description: The results of applying the best-performing ternary sentiment analysis model to the 20,000-row sample dataset.

### Jupyter Notebook Files
1. **SentimentLexicon(AFINN&VADER)_Twoclass.ipynb**
   - Description: Testing AFINN and VADER sentiment analysis models using binary labeled sentiment analysis data.

2. **SentimentLexicon(AFINN&VADER)_Threeclass.ipynb**
   - Description: Testing AFINN and VADER sentiment analysis models using ternary labeled sentiment analysis data.

3. **ML(KNN,RF,SVM,ANN,XGBoost,NB)_TF_IDF_TwoClasses.ipynb**
   - Description: Training and testing six different traditional machine learning models using binary labeled sentiment analysis data.

4. **ML(KNN,RF,SVM,ANN)_TF_IDF_ThreeClasses.ipynb**
   - Description: Training and testing four different traditional machine learning models using ternary labeled sentiment analysis data.

5. **ML(XGBoost,NB)_TF_IDF_ThreeClasses.ipynb**
   - Description: Training and testing two different traditional machine learning models using ternary labeled sentiment analysis data.

6. **TL(SieBERT&RoBERTa)_TwoClass.ipynb**
   - Description: Training and testing transfer learning models SieBERT and RoBERTa using binary labeled sentiment analysis data.

7. **TL(SieBERT&RoBERTa)_ThreeClass.ipynb**
   - Description: Training and testing transfer learning models SieBERT and RoBERTa using ternary labeled sentiment analysis data.

8. **Combined_TM(Full_dataset).ipynb**
   - Description: Performing topic modeling on the full 20,000-row dataset.

9. **Combined_TM(negative).ipynb**
   - Description: Performing topic modeling on the negative sentiment dataset.

10. **Combined_TM(neutral).ipynb**
    - Description: Performing topic modeling on the neutral sentiment dataset.

11. **Combined_TM(positive).ipynb**
    - Description: Performing topic modeling on the positive sentiment dataset.

12. **Apply_model_to_data_&_Visualization.ipynb**
    - Description: Applying the best-performing ternary sentiment classification model, RoBERTa, to the entire 20,000-row dataset (divided into four subsets) and performing exploratory analysis on project success rates and sentiment distribution.

13. **Predictive_Model.ipynb**
    - Description: Training predictive models using different machine learning models, comparing accuracy and other metrics, and analyzing the importance of different variables (including sentiment variables extracted from the text) in predicting project success.

## How to Run the Code
1. Clone this repository to your local machine:
   ```sh
   git clone https://github.com/your-username/your-repository.git
   
## Dependencies
- **pandas**: For data manipulation and analysis.
- **numpy**: For scientific computing and array operations.
- **scikit-learn**: For training and evaluating machine learning models, including preprocessing, model selection, and metrics.
- **statsmodels**: For statistical modeling and econometrics analysis.
- **datasets**: For handling and managing large datasets, particularly useful in conjunction with Hugging Face's `transformers`.
- **nltk**: For natural language processing tasks such as tokenization, stopwords removal, and more.
- **gensim**: For topic modeling and word vector representation.
- **contextualized_topic_models**: For context-aware topic modeling that combines contextual embeddings with traditional topic models.
- **transformers**: For handling and using pre-trained deep learning models from Hugging Face, especially for tasks like sentiment analysis.
- **afinn**: For dictionary-based sentiment analysis, providing a simple way to analyze text sentiment.
- **vaderSentiment**: For VADER sentiment analysis, which is particularly good for social media texts.
- **matplotlib**: For data visualization, creating static, animated, and interactive visualizations.
- **seaborn**: For advanced data visualization based on matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.
- **torch**: For training and inference of deep learning models, part of the PyTorch library.
- **xgboost**: For gradient boosting tree models, widely used for structured/tabular data.

## Acknowledgements
This project was inspired by and benefited from the following GitHub repositories:

- [MilaNLProc/contextualized-topic-models](https://github.com/MilaNLProc/contextualized-topic-models)
- [j-hartmann/siebert](https://github.com/j-hartmann/siebert)

We would like to thank the authors of these projects for their valuable contributions to the open-source community.

