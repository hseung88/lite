from lite.methods.common.data import PredictiveMarginals
from lite.methods.lite.posterior import predict_marginals


class LITEPredictor:
    name = "LITE"

    def __init__(self, m, prediction_batch_size=256, *, normalize_directions=False):
        self.m, self.prediction_batch_size = m, prediction_batch_size
        self.normalize_directions = normalize_directions
        self.data = None

    def build(self, data):
        self.data = data

    def predict_f_marginals(self, X_eval, *, neighborhoods=None, return_expected_mse=False):
        if self.data is None:
            raise RuntimeError("build() must precede prediction")
        data = self.data
        result = predict_marginals(
            X_eval,
            data.X_train,
            data.f_train_obs,
            data.g_train_obs,
            lengthscale=data.lengthscale,
            outputscale=data.outputscale,
            noise_y_var=data.sigma_f**2,
            noise_g_var=data.sigma_g**2,
            kernel=data.kernel_name,
            gradient_noise_model="iid",
            m=self.m,
            prediction_batch_size=self.prediction_batch_size,
            train_scaled=data.X_train_scaled,
            neighborhoods=neighborhoods,
            normalize_directions=self.normalize_directions,
            return_expected_mse=return_expected_mse,
        )
        return PredictiveMarginals(
            mean=result[0], var=result[1], expected_mse=result[2] if return_expected_mse else None
        )
