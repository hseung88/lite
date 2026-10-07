from lite.methods.lite.model import LITEModel as _Model

from .validation import MD22Validation


class LITEModel(MD22Validation, _Model):
    def __init__(self, *args, normalize_directions=True, **kwargs):
        super().__init__(*args, normalize_directions=normalize_directions, **kwargs)
