class TokenLengthError(Exception):
    """Собственное исключение."""

    def __init__(self, message="Количество используемых токенов больше допустимого"):
        self.message = message
        super().__init__(self.message)
