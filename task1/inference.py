import pandas as pd

from utils.processing import clean_text


def get_predict(df: pd.DataFrame, vectorizer, model) -> pd.DataFrame:
    try:
        ucid = df.ucid
    except:
        print('Не хватает столбца ucid')
    else:
        try:
            X = df.text_employer
        except:
            print('Не хватает столбца text_employer')
        else:
            # # Очистка текста
            documents = X.apply(clean_text)

            # TF-IDF
            X = vectorizer.transform(documents.values).toarray()

            # Предикт
            y_pred = model.predict(X)

            report = {'ucid': ucid, 'message': df.text_employer, 'predict': y_pred}
            return pd.DataFrame(report)
