import joblib
import pandas as pd

from utils import clean_text


def get_predict(df):
    vectorizer = joblib.load('./models/tfidfconverter.pkl')
    model = joblib.load('./models/randomforest.pkl')

    try:
        ucid = df.ucid
    except:
        print('Не хватает столбца ucid')
    else:
        try:
            X, y = df.text_employer, df.ACTION_ITEM_RESULT_PRODUCT_NAME
        except:
            print('Не хватает столбцов text_employer или ACTION_ITEM_RESULT_PRODUCT_NAME')
        else:
            # Очистка текста
            documents = X.apply(clean_text)

            # TF-IDF
            X = vectorizer.transform(documents.values).toarray()

            # Предикт
            y_pred = model.predict(X)

            report = {'ucid': ucid, 'message': df.text_employer, 'predict': y_pred}
            return pd.DataFrame(report)
