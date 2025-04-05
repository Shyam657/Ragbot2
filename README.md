# Deploying Your RAG Application to Render.com

This guide will walk you through converting your Colab notebook into a Streamlit web application and deploying it on Render.com.

## 1. Project Structure

Create the following files in your GitHub repository:

```
rag-app/
├── app.py             # Main Streamlit application
├── requirements.txt   # Dependencies
├── utils.py           # Helper functions
├── .gitignore         # Git ignore file
└── README.md          # Documentation
```

## 2. Set Up Files

### app.py
This is your main Streamlit application that will handle the user interface.

### requirements.txt
List all the packages your application needs:

```
streamlit
langchain
langchain-community
langchain-core
langchain-together
langchain-chroma
langchain-text-splitters
arxiv
pymupdf
python-dotenv
chromadb
matplotlib
pdfminer.six
```

### .gitignore
```
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
chroma_db/
*.db
```

## 3. Deploy to Render.com

1. Create an account on Render.com
2. Create a new Web Service
3. Connect to your GitHub repository
4. Configure:
   - Name: `rag-application` (or your preferred name)
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py`
   - Environment Variables:
     - Set up your API keys: `TOGETHERAI_API_KEY`, `LANGCHAIN_API_KEY`, etc.
   - Plan: Start with the free tier for testing

5. Click "Create Web Service"

## 4. Verify Deployment
After deployment completes, Render will provide a URL where your application is hosted.
