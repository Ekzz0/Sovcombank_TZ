import re
import pymorphy3
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(data, stopwords=stopwords.words('russian')):
    text = data.lower()
    text = re.sub(r'[^\sа-яА-Яa-zA-Z0-9@\[\]]', ' ', text)  # Удаляет пунктцацию
    text = re.sub(r'\w*\d+\w*', '', text)  # Удаляет цифры
    text = re.sub(r'\s{2,}', " ", text)  # Удаляет ненужные пробелы
    text = re.sub(r'[\n\r]', "", text)  # Удаляет ненужные пробелы
    # text = re.sub(r'_+', ' ', text)  # Удалим _
    text = [w for w in text.split() if w not in stopwords]  # Удалим стоп-слова

    return ' '.join(text)


def lemmatization_text(data, morph=pymorphy3.MorphAnalyzer()):
    result = ' '.join([morph.parse(x)[0].normal_form for x in data.split()])
    return result


# Создадим функцию очистки текста
def clean_text(data):
    try:
        result = preprocess_text(data)
        result = lemmatization_text(result)
    except:
        return ''
    else:
        return result