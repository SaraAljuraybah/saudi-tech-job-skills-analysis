# Saudi Tech Role Predictor App

This folder adds a Streamlit web application to the existing **Saudi Tech Job Skills Analysis** project.

The app allows a user to enter skills such as:

```text
Python, SQL, Machine Learning, Power BI, Data Visualization, Statistics
```

Then it predicts the most suitable role from:

- AI/ML Engineer
- Data Analyst
- Data Engineer
- Data Scientist

## What the app does

The app extends the modelling notebook into a working web interface.

The user enters skills as text. The app converts the text into TF-IDF numerical features, sends those features to the trained Random Forest classifier, and displays:

- Recommended role
- Confidence score
- Role matching scores for all classes
- A short explanation of the recommendation

## How the model works

The original modelling notebook uses:

- `clean_description` as the text feature
- `skills_text` as the renamed modelling input
- `role_label` as the target label
- TF-IDF vectorization with:
  - `max_features=3000`
  - `ngram_range=(1, 2)`
  - `stop_words='english'`
- Random Forest as the best model based on accuracy and weighted F1-score

## Files in this folder

```text
app/
├── app.py
├── train_and_save_model.py
├── requirements.txt
└── README_app.md
```

After training, the script also creates:

```text
app/
├── role_model.pkl
└── tfidf_vectorizer.pkl
```

## How to install requirements

From the project root or from the app folder, run:

```bash
pip install -r app/requirements.txt
```

If you are already inside the app folder:

```bash
pip install -r requirements.txt
```

## How to train and save the model

From the project root:

```bash
cd app
python train_and_save_model.py
```

This creates:

```text
role_model.pkl
tfidf_vectorizer.pkl
```

## How to run the app

After training:

```bash
streamlit run app.py
```

The app will open in your browser.

## Example input

```text
Python, SQL, Machine Learning, Pandas, Data Visualization, Statistics
```

Possible output:

```text
Recommended Role: Data Scientist
Confidence Score: 82%
```

The exact score may differ depending on the dataset and model training result.

## Troubleshooting

### Missing model files

Error:

```text
Missing required file(s): role_model.pkl, tfidf_vectorizer.pkl
```

Fix:

```bash
python train_and_save_model.py
```

### Dataset not found

Make sure the app folder is inside the project root:

```text
saudi-tech-job-skills-analysis/
├── data/
│   ├── jobs_sa_model_ready.csv
│   └── jobs_sa_cleaned.csv
└── app/
    ├── app.py
    └── train_and_save_model.py
```

### Stratify error during train/test split

This happens when one role has too few samples. The script checks for this and prints class counts. Make sure the dataset contains enough examples for each of the four target roles.

### Wrong prediction results

The prediction depends on the training dataset. For best results, enter clear skills related to AI/Data roles, not full sentences with unrelated text.

## Presentation script

Our project analyzes Saudi tech job postings to understand the skills required for different AI and data roles. As an extension, we built a machine learning web application that predicts the most suitable role based on user-entered skills. The input text is transformed using TF-IDF, then passed to a trained classification model. The model outputs the recommended role and confidence score.
