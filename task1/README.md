## Задание 1. Классификация текста

Набор данных ```../topic_modeling_task_sample_trainPart.csv```, представляющий собой транскрибацию реплик менеджера в диалоге с клиентом по продаже продукта. В датасете в поле "text_employer" одна строка представляет собой один диалог, где в качестве текста выступают соединенные через точку все реплики менеджера из одного диалога.
В файле есть поле "ACTION_ITEM_RESULT_PRODUCT_NAME". в нем содержаться темы, к которым относились данные диалоги.
Необходимо создать модель которая будет предсказывать по тексту к какой из тем относился конкретный диалог. Готовая модель должна на вход получать сроку и возвращать код (1-4):
1. Бизнес-карта
2. Зарплатные проекты
3. Открытие банковского счета
4. Эквайринг

Конечный результат:
1. ноутбук с решением (включающий итоговые метрики точности, confusion matrix)
2. таблица со столбцами: ucid, порядковый номер реплики, реплика менеджера, код (1-4).
3. ноутбук с функцией или .py файл, на вход который может принять файл .csv и вывести результаты предсказания в виде таблицы или файла

## Решение:
1) Изначально в ```../data_preparation.ipnyb``` была произведена обработка текста:
    - Нижний регистр (**lower()**)
    - Удаление пунктуации (**re**)
    - Удаление цифр (**re**)
    - Удаление ненужных пробелов (**re**)
    - Удаление стоп-слов (**nltk**)
    - Лемматизация (**pymorphy3**)
2) Был использован RandomUnderSampler для того, чтобы сбалансировать классы целевой переменной
3) Обучен TfidfVectorizer
4) Обучена модель RandomForestClassifier

**Результаты валидации модели на тестовой выборке:**

|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.95      | 0.79   | 0.86     | 970    |
| 1            | 0.87      | 0.77   | 0.82     | 637    |
| 2            | 0.55      | 0.88   | 0.68     | 319    |
| 3            | 0.78      | 0.95   | 0.85     | 205     |
| accuracy     |           |        | 0.81     | 2131    |
| macro avg    | 0.79      | 0.85   | 0.80     | 2131    |
| weighted avg | 0.88      | 0.81   | 0.82     | 2131    |

**Confusion Matrix:**

!(conf_matr)[./c_matrix.png]

### Для получение прогноза по сырому файлу используется файл ```inference.py```