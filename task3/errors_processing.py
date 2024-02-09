from typing import Callable
from task3.my_errors import TokenLengthError
from config import MAX_NUM_OF_TOKENS
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


# Метод для валидации количества токенов
def token_validation(message: str) -> None:
    messages = [
        HumanMessage(content=f"{message}")
    ]
    chat = ChatOpenAI(temperature=0)
    num_of_tokens = chat.get_num_tokens_from_messages(messages)

    if num_of_tokens > MAX_NUM_OF_TOKENS:
        raise TokenLengthError


# Декоратор для обработки собственных ошибок
def errors_handler(func: Callable) -> Callable:
    def wrapper(*args, **kwargs) -> Callable:
        try:
            token_validation(args[0])
            return func(*args, **kwargs)
        except TokenLengthError as e:
            print(e.message)
        except Exception:
            print("Неизвестная ошибка!")

    return wrapper
