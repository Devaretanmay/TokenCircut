class TokenCircuitError(RuntimeError):
    pass


class StateStagnationError(TokenCircuitError):
    pass


class FutileActionError(TokenCircuitError):
    pass
