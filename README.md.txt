================================================
         APPLICATION FILES - REQUIRED TO RUN
================================================

This folder contains all files necessary to run the
Smart Resume Screening Application.

================================================
FILES IN THIS FOLDER:
================================================

1. resume_screening.py (13 KB)
   - Main application code
   - Contains all UI and processing logic

2. clf.pkl (601 KB)
   - Trained Multinomial Naive Bayes classifier
   - Predicts job categories from resume text

3. tfidf.pkl (65 KB)
   - TF-IDF vectorizer model
   - Converts text to numerical features

4. category_mapping.pkl (596 bytes)
   - Maps category IDs to job names
   - 25 job categories supported

5. requirements.txt (105 bytes)
   - Lists all Python package dependencies
   - Used by: pip install -r requirements.txt

================================================
HOW TO RUN:
================================================

Step 1: Create Virtual Environment
python3 -m venv venv
source venv/bin/activate

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Run Application
streamlit run resume_screening.py

Step 4: Access Application
Browser opens automatically at http://localhost:8501

================================================
SYSTEM REQUIREMENTS:
================================================

- Python 3.8 or higher
- 2 GB RAM minimum
- 500 MB disk space
- Internet (for initial package installation)
- Operating System: Windows, macOS, or Linux

================================================
ALL FILES IN THIS FOLDER ARE REQUIRED
================================================

Do not delete any file. The application will not
run without all 5 files present.

================================================
