from .model import TERAModel as _TERAModel
from .model import TERAPredictor as TERAPredictor
from .model import _alpha_from_r as _alpha_from_r
from .model import _beta_from_r as _beta_from_r
from .model import _direct_joint_scalar_conditional as _direct_joint_scalar_conditional
from .model import _func_covariance as _func_covariance
from .model import (
    _local_observed_y_factor as _factor,
)
from .model import _projected_gradient_noise_gram as _projected_gradient_noise_gram


class TERAModel(_TERAModel):
    def _batch_nll(self, **kwargs):
        from .model import _batch_nll

        return _batch_nll(training_mode=self.training_mode, gram_distances=True, **kwargs)


class SequentialTERAModel(TERAModel):
    """Evaluate BO training factors one at a time within each optimizer batch."""

    def _batch_nll(self, **kwargs):
        from .model import _batch_nll_sequential

        return _batch_nll_sequential(
            training_mode=self.training_mode, gram_distances=True, **kwargs
        )


def _local_observed_y_factor(**kwargs):
    return _factor(gram_distances=True, **kwargs)
