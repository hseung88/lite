#!/usr/bin/env python3
import warnings

import torch
from gpytorch import settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.settings import trace_mode
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.utils.errors import CachingError
from gpytorch.utils.memoize import cached, clear_cache_hook, pop_from_cache_ignore_args
from gpytorch.utils.warnings import OldVersionWarning
from gpytorch.variational._variational_strategy import _VariationalStrategy
from linear_operator import to_dense as delazify
from linear_operator.operators import (
    DiagLinearOperator as DiagLazyTensor,
)
from linear_operator.operators import (
    MatmulLinearOperator as MatmulLazyTensor,
)
from linear_operator.operators import (
    RootLinearOperator as RootLazyTensor,
)
from linear_operator.operators import (
    SumLinearOperator as SumLazyTensor,
)
from linear_operator.operators import (
    TriangularLinearOperator as TriangularLazyTensor,
)


def _ensure_updated_strategy_flag_set(
    state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
):
    device = state_dict[list(state_dict.keys())[0]].device
    if prefix + "updated_strategy" not in state_dict:
        state_dict[prefix + "updated_strategy"] = torch.tensor(False, device=device)
        warnings.warn(
            "You have loaded a variational GP model (using `VariationalStrategy`) from a previous version of "
            "GPyTorch. We have updated the parameters of your model to work with the new version of "
            "`VariationalStrategy` that uses whitened parameters.\nYour model will work as expected, but we "
            "recommend that you re-save your model.",
            OldVersionWarning,
        )


class DirectionalGradVariationalStrategy(_VariationalStrategy):
    def __init__(
        self,
        model,
        inducing_points,
        inducing_directions,
        variational_distribution,
        learn_inducing_locations=True,
    ):
        super().__init__(model, inducing_points, variational_distribution, learn_inducing_locations)
        self.register_buffer("updated_strategy", torch.tensor(True))
        self._register_load_state_dict_pre_hook(_ensure_updated_strategy_flag_set)
        self.register_parameter(
            name="inducing_directions", parameter=torch.nn.Parameter(inducing_directions.clone())
        )

    @cached(name="cholesky_factor", ignore_args=True)
    def _cholesky_factor(self, induc_induc_covar):
        L = psd_safe_cholesky(
            delazify(induc_induc_covar).double(),
            jitter=settings.cholesky_jitter.value(dtype=induc_induc_covar.dtype),
        )
        return TriangularLazyTensor(L)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        zeros = torch.zeros(
            self._variational_distribution.shape(),
            dtype=self._variational_distribution.dtype,
            device=self._variational_distribution.device,
        )
        ones = torch.ones_like(zeros)
        res = MultivariateNormal(zeros, DiagLazyTensor(ones))
        return res

    def forward(
        self, x, inducing_points, inducing_values, variational_inducing_covar=None, **kwargs
    ):

        kwargs.pop("diag", None)

        inducing_directions = self.inducing_directions
        derivative_directions = kwargs["derivative_directions"]

        num_induc = inducing_points.size(-2)
        num_directions = int(inducing_directions.size(-2) / num_induc)
        num_data = x.size(-2)
        num_derivative_directions = int(derivative_directions.size(-2) / num_data)
        assert num_derivative_directions == num_directions, (
            "Need minibatch dim to be same as number of directions for kernel"
        )

        full_inputs = torch.cat([inducing_points, x], dim=-2)

        test_mean = self.model.mean_module(
            x.repeat_interleave(num_derivative_directions + 1, dim=0)
        )

        kwargs["v1"] = inducing_directions.to(x.device)
        kwargs["v2"] = derivative_directions.to(x.device)
        self.model.covar_module.base_kernel.set_num_directions(num_directions)
        full_output = self.model.covar_module(inducing_points, x, **kwargs)
        induc_data_covar = full_output.to_dense()
        kwargs["v1"] = derivative_directions.to(x.device)
        kwargs["v2"] = inducing_directions.to(x.device)
        self.model.covar_module.base_kernel.set_num_directions(num_directions)
        full_output = self.model.covar_module(x, inducing_points, **kwargs)
        data_induc_covar = full_output.to_dense()

        kwargs["v1"] = inducing_directions.to(x.device)
        kwargs["v2"] = inducing_directions.to(x.device)
        self.model.covar_module.base_kernel.set_num_directions(num_directions)
        full_output = self.model.forward(inducing_points, **kwargs)
        induc_induc_covar = full_output.lazy_covariance_matrix.add_jitter()
        kwargs["v1"] = derivative_directions.to(x.device)
        kwargs["v2"] = derivative_directions.to(x.device)
        self.model.covar_module.base_kernel.set_num_directions(num_directions)
        full_output = self.model.forward(x, **kwargs)
        data_data_covar = full_output.lazy_covariance_matrix

        L = self._cholesky_factor(induc_induc_covar)
        if L.shape != induc_induc_covar.shape:
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(induc_induc_covar)
        interp_term = L.solve(induc_data_covar.double()).to(full_inputs.dtype)

        interp_term_trans = L.solve(data_induc_covar.transpose(-1, -2).double()).to(
            full_inputs.dtype
        )

        predictive_mean = (
            interp_term_trans.transpose(-1, -2) @ inducing_values.unsqueeze(-1)
        ).squeeze(-1) + test_mean

        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = SumLazyTensor(variational_inducing_covar, middle_term)

        if trace_mode.on():
            predictive_covar = (
                data_data_covar.add_jitter(1e-4).to_dense()
                + interp_term_trans.transpose(-1, -2) @ middle_term.to_dense() @ interp_term
            )
        else:
            predictive_covar = SumLazyTensor(
                data_data_covar.add_jitter(1e-4),
                MatmulLazyTensor(interp_term_trans.transpose(-1, -2), middle_term @ interp_term),
            )

        return MultivariateNormal(predictive_mean, predictive_covar)

    def __call__(self, x, prior=False, **kwargs):
        if not self.updated_strategy.item() and not prior:
            with torch.no_grad():
                prior_function_dist = self(self.inducing_points, prior=True)
                prior_mean = prior_function_dist.loc
                L = self._cholesky_factor(prior_function_dist.lazy_covariance_matrix.add_jitter())

                orig_mean_init_std = self._variational_distribution.mean_init_std
                self._variational_distribution.mean_init_std = 0.0

                variational_dist = self.variational_distribution
                mean_diff = (variational_dist.loc - prior_mean).unsqueeze(-1).double()
                whitened_mean = L.solve(mean_diff).squeeze(-1).to(variational_dist.loc.dtype)
                covar_root = (
                    variational_dist.lazy_covariance_matrix.root_decomposition()
                    .root.to_dense()
                    .double()
                )
                whitened_covar = RootLazyTensor(L.solve(covar_root).to(variational_dist.loc.dtype))
                whitened_variational_distribution = variational_dist.__class__(
                    whitened_mean, whitened_covar
                )
                self._variational_distribution.initialize_variational_distribution(
                    whitened_variational_distribution
                )

                self._variational_distribution.mean_init_std = orig_mean_init_std

                clear_cache_hook(self)

                self.updated_strategy.fill_(True)

        return super().__call__(x, prior=prior, **kwargs)
