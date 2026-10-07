from lite.methods.standard.model import StandardGPModel as _Model

from .validation import MD22Validation


class StandardGPModel(MD22Validation, _Model):
    pass
