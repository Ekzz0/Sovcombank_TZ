## Задание 2. Тематическое моделирование
Подразумевается, что в репликах менеджера (поле "text_employer") прослеживаются такие топики:
- Приветствие
- Перенос активности
- Окончание разговора
- "Болтовня"
- Следующие шаги
- Безопасность
- Тарифы / Цены
- Акции
- Удобство работы
- Простота настройки
- Скорость работы
- Преимущества
- Вовлечение (подведение к диалогу)
- Призыв к действию
- Инструкция по подключению/приобретению
- Вопрос клиента

Необходимо провести тематическое моделирование, приблизив итоговый набор тем модели к данному набору топиков.

Конечный результат:
1. ноутбук с решением
2. таблица со столбцами ucid, порядковый номер реплики, реплика менеджера, топик (т.е. топик должен быть получен для каждой реплики)

## Решение:
1) При помощи модуля ```gensim``` была построена ```LdaModel```
2) Посчитана сложность модели и когеренция
3) При помощи ``pyLDAvis``  отображено разделение сообщений на топики
4) Написан метод ``get_topic`` для получения значения топика по сообщению
5) Отображена таблица указанного в задании формата

## Топики:
При помощи ChatGPT 3.5 было выделено соответствие получившихся топиков и указанных в ТЗ тем:

1. **Приветствие**: Топик 11 ("добрый день", "звонить", "удобно").
2. **Перенос активности**: Топик 15 ("перезвонить", "время", "связаться").
3. **Окончание разговора**: Топик 14 ("вопрос", "ваш", "помочь").
4. **"Болтовня"**: Топик 5 ("просто", "это", "именно").
5. **Следующие шаги**: Топик 13 ("перевод", "мочь", "подключить").
6. **Безопасность**: Топик 8 ("лимит", "банк", "безопасность").
7. **Тарифы / Цены**: Топик 12 ("счёт", "расчётный", "тариф").
8. **Акции**: Нет явного соответствия.
9. **Удобство работы**: Топик 6 ("карта", "отделение", "обслуживание").
10. **Простота настройки**: Топик 10 ("карта", "наличный", "предложение").
11. **Скорость работы**: Нет явного соответствия.
12. **Преимущества**: Топик 4 ("контакт", "удалой", "свой").
13. **Вовлечение (подведение к диалогу)**: Топик 7 ("тысяча", "рубль", "комиссия").
14. **Призыв к действию**: Топик 9 ("терминал", "эквайринг", "установка").
15. **Инструкция по подключению/приобретению**: Топик 2 ("почта", "номер", "указать").
16. **Вопрос клиента**: Топик 0 ("пожалуйста", "линия", "вопрос").