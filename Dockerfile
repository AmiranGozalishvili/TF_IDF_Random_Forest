
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
RUN apt-get update && apt-get install build-essential -y

RUN pip install --upgrade pip
RUN pip install -U spacy
RUN pip install pandas
RUN pip install contractions
RUN pip install ipywidgets
RUN pip install -U nltk
RUN pip install autocorrect
RUN pip install openpyxl -U
RUN python3 -m spacy download en_core_web_sm
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install textblob
RUN pip install wordcloud
RUN pip install sklearn
RUN python -c "import nltk; nltk.download('wordnet')"
RUN python -c "import nltk; nltk.download('punkt')"
RUN python -c "import nltk; nltk.download('stopwords')"

#RUN pip install textattack
#RUN pip install tensorflow_text


COPY ./app /app

WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
