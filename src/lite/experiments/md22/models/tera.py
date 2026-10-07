from lite.methods.tera.model import TERAModel as _Model
from lite.methods.tera.model import _batch_nll_sequential

from .validation import MD22Validation


class TERAModel(MD22Validation, _Model):
    """Sequential factor evaluation and prediction, with minibatch Adam updates."""

    def _batch_nll(self, **kwargs):
        return _batch_nll_sequential(training_mode=self.training_mode, **kwargs)

    def _make_predictor(self):
        self.prediction_batch_size = 1
        return super()._make_predictor()


class BatchedTERAModel(MD22Validation, _Model):
    """Batched factor evaluation and configurable prediction batches."""

    pass
