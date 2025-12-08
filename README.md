# **Phase 4 Project: NLP Sentiment Analysis of Apple vs Google Tweets**



# 1. **Executive Summary**

This project builds a sentiment classification system to analyze public emotions in Apple-related tweets. Using a dataset of 9,093 tweets labeled as positive, negative, or neutral, the goal is to help Apple’s Marketing & Product teams better understand customer perception and identify emerging issues in real time. Tweets are short, informal, and often noisy, making NLP-based machine learning an appropriate solution.

Data preparation included lowercasing, removing URLs, mentions, emojis, punctuation, and extra whitespace, followed by stopword removal and lemmatization to reduce noise. Duplicates, missing labels, and extremely short tweets were removed to ensure training quality. TF-IDF vectorization powered the classical models, while HuggingFace Transformers handled the BERT-based approaches.

Several models were evaluated. Traditional machine learning classifiers (Logistic Regression, Naive Bayes, Linear SVM, SMOTE-SVM, and Weighted SVM) achieved accuracies between 63% and 70%, with weighted SVM performing the strongest among them. A fine-tuned DistilBERT model significantly outperformed classical models, achieving 71.16% accuracy with 2 epochs and 71.49% accuracy after 4 epochs—confirming the advantage of transformer models for short-form, informal text like tweets.

Model evaluation used an 80/20 train-test split, accuracy, weighted F1-score, and confusion matrices. DistilBERT was selected as the final model due to its superior performance, ability to capture context, and improved handling of neutral vs. positive/negative distinctions. Further fine-tuning and domain-specific training data can boost accuracy even more.ults.



 # 2.  **Business Understanding**

 # 3.  **Stakeholder**  

**Primary Stakeholders**

1. Apple Marketing Team
Uses real-time sentiment insights to understand public perception, monitor brand reputation, evaluate campaign performance, and respond quickly to public feedback after product launches or events.

2. Apple Product Management & Engineering Teams
Gain visibility into customer frustrations, feature requests, recurring complaints, and performance issues (e.g., crashes, battery problems). This helps prioritize product improvements and software patches.

**Secondary Stakeholders**

3. Apple Customer Support & Care Teams
Benefit from early detection of negative sentiment spikes related to device malfunctions or software bugs. They can proactively prepare responses, FAQs, or issue statements.

4. Apple PR & Corporate Communications
Use sentiment monitoring to manage crises, track public reaction during controversies, and craft timely communication strategies compared to competitors like Google.

5. Competitive Intelligence & Market Research Teams
Analyze Apple vs Google sentiment trends to understand market positioning, customer loyalty drivers, and areas where Apple is outperforming or lagging behind.


     
 # 4. **Business problem**
    
Apple’s Marketing & Product Intelligence teams require a system that can continuously monitor and compare public sentiment toward Apple products against competing brands—particularly Google. With social media platforms like Twitter influencing customer perception in real time, Apple needs early visibility into negative trends, emerging complaints, and user frustrations before they scale. Likewise, understanding which Apple products generate the strongest positive engagement—compared to similar Google offerings—provides insights for campaign optimization, product refinement, and competitive positioning.

# 4,1 **Project Objectives**
This project aims to build an NLP-based sentiment analysis model capable of automatically classifying Apple- and Google-related tweets as positive, negative, or neutral. The solution will help Apple quantify customer emotions, detect shifts in public perception, and benchmark Apple’s reputation relative to Google in fast-evolving online conversations.

# 5. **Data Understanding**

The dataset comes from CrowdFlower (hosted on data.world) and contains 9,093 tweets related to Apple products and competing brands. It includes three columns:

1. tweet_text – the raw text of each tweet

2. emotion_in_tweet_is_directed_at – the specific product or brand mentioned (e.g., iPad, Apple, Google)

3. is_there_an_emotion_directed_at_a_brand_or_product – the sentiment label assigned by annotators

Sentiment labels fall into four categories: Positive emotion, Negative emotion, No emotion toward brand or product, and I can’t tell. The distribution is imbalanced: most tweets show no emotion (5,389), followed by positive (2,978), negative (570), and uncertain (156).

Basic descriptive statistics show an average tweet length of ~105 characters, with lengths ranging from 3 to 178 characters, confirming that the dataset fits typical short-text social media patterns. The column emotion_in_tweet_is_directed_at contains multiple Apple and non-Apple target entities, with the most common being iPad (946), Apple (661), and iPad/iPhone apps (470).

This dataset is well-suited for an NLP sentiment classification task because it provides labeled text, diverse sentiment categories, and real-world social media noise—allowing meaningful training and evaluation of advanced models such as DistilBERT.


# **Models Results and Recommendation**

    
![Model Accuracy Comparison](images/output_84_0.png)
    



    
![Accuracy vs Weighted F1 for All Models](images/output_84_1.png)
    

This project evaluated a range of machine-learning and transformer-based models for sentiment analysis on apple_google_twitter data. The goal was to identify the model that offers the best balance of accuracy, robustness to class imbalance, and real-world practicality. Seven models were compared, including classical machine-learning baselines, SMOTE-enhanced variants, and two versions of a fine-tuned DistilBERT transformer model.

1. **Overall Model Performance**

Across all experiments, transformer-based models (DistilBERT) consistently outperformed traditional machine-learning approaches.
The 4-epoch DistilBERT model achieved the highest performance, with:

* Test Accuracy: **0.7149**

* Strong loss reduction over epochs **(0.78 to 0.34)**

* Excellent stability and generalization on unseen data

This confirms the advantage of pretrained contextual embeddings for sentiment tasks, especially when dealing with the informal, context-rich nature of Twitter language.

Following DistilBERT, the Weighted SVM model emerged as the strongest classical approach, achieving:

* Accuracy: 0.6981

* Competitive Weighted F1: 0.69

* Improved recall on the minority class without overly sacrificing the majority class

This demonstrates that class-weighted optimization effectively compensates for label imbalance, a core challenge of real-world sentiment datasets.

2. Comparative Strengths and Weaknesses

* Models such as Naive Bayes and SVM provided respectable baselines, but their inability to capture semantic nuance limited performance. Logistic
* Regression significantly underperformed due to linear decision boundaries that cannot adequately model complex emotional expressions.
* The SMOTE-SVM model successfully improved minority-class recall but at the cost of overall accuracy, showing that synthetic oversampling must be       applied carefully to avoid feature-space distortion.

* The transformer-based DistilBERT models, however, demonstrated:

* Superior semantic understanding

* Strong contextual sensitivity

* Better performance on ambiguous or indirect sentiment expressions

* Minimal preprocessing requirements. These advantages were especially evident in class 2 (positive sentiment), where BERT-based models achieved industry-standard recall levels.

# 8.1 **Final Model Recommendation**

After evaluating all performance metrics, computational costs, and robustness considerations, the DistilBERT (4-epoch) model is recommended as the final production model for Apple tweet sentiment analysis. 
It:

1. Achieves the highest accuracy

2. Demonstrates reliable convergence behavior

3. Handles class imbalance more effectively than baselines

4. Captures subtle sentiment cues essential in social-media data

This makes it the optimal choice for both real-time monitoring and analytical applications involving brand perception, customer satisfaction, and public opinion tracking.

# 9. **Recommendations to Stakeholder**

Based on the sentiment analysis findings and model performance results, we recommend the following actions for the Marketing & Product Team at Apple:

1. Deploy the BERT Model to Monitor Real-Time Customer Sentiment

BERT achieved the highest accuracy (0.7149) and consistently captured nuanced language in tweets.
Deploying this model in production will give Apple reliable real-time insight into public sentiment trends across product lines.

2. Prioritize Investigation of Negative Sentiment Mentions

All models found negative sentiment to be the smallest but hardest class to predict, suggesting customers express negative feedback in subtle or indirect ways.
Apple should implement alerts when negative sentiment spikes so the team can address emerging issues before they escalate.

3. Break Down Sentiment by Product Category

Transformer models capture context well—use this strength to segment sentiment by:

* iPhone

* Mac

* iPad

* Accessories

* Software & Services
This enables more targeted product improvements and campaign adjustments.

4. Use Sentiment Insights to Enhance Customer Engagement Campaigns

The large volume of neutral and positive tweets indicates opportunities to: 

* amplify positive product experiences

* convert neutral conversations into loyalty-building interactions

* create targeted messaging where sentiment is weak or shifting

5. Continuously Retrain the Model With New Tweet Data

Because customer sentiment evolves quickly, especially after product launches or major events, the model should be retrained quarterly or after key Apple announcements to maintain accuracy.

# 10. **Limitations**

* Neutral tweets are often ambiguous.
Many tweets classified as neutral contain mixed or unclear sentiment, making them difficult for even advanced models to interpret accurately. This affects overall precision, especially when distinguishing between mild positive or mild negative opinions.

* Tweets are short and highly contextual.
Twitter posts often include slang, abbreviations, emojis, and cultural references. These characteristics reduce the amount of linguistic information available, making it challenging for models to fully capture meaning.

* Sarcasm detection remains difficult.
Sentiment expressed sarcastically (e.g., “Great, my iPhone died again 🙄”) is hard for machine learning models to interpret correctly. This leads to misclassification in negative sentiment cases.

* Model trained on historical data.
The model reflects past patterns of customer language and may not immediately adapt to new product releases, evolving slang, or emerging sentiment shifts. Regular retraining with fresh Apple-related tweets is necessary.

* Imbalanced sentiment classes.
Negative tweets were the smallest portion of the dataset, which reduced model performance on this class despite techniques like class weights and SMOTE.

# 11. **Future Work**

* Fine-tune a transformer model further.
Increasing training epochs, expanding the dataset, and adjusting learning rates can significantly improve BERT-based performance—especially for neutral and sarcastic tweets.

* Collect more and fresher Apple-specific tweets.
Expanding the dataset with recent product launches (iPhone, Mac, iPad, Vision Pro) will help the model stay current with evolving customer language and sentiment trends.

* Implement time-series sentiment tracking.
Monitoring sentiment over weeks, months, or around product events (e.g., WWDC, product launches) would allow Apple to detect shifts earlier and respond quickly.

* Domain-specific lexicon enhancement.
Incorporating Apple-related terms (e.g., “FaceID,” “AirDrop,” “iOS update,” “battery health”) can improve contextual understanding and reduce misclassifications.

* Build a real-time dashboard.
A live system visualizing positive, neutral, and negative sentiment would give the marketing team actionable insights during major announcements or crises.
