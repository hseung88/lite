from lite.experiments.md22.models.derivative_baselines import DerivativeBaselineModel


class DSoftKIModel(DerivativeBaselineModel):
    def __init__(self, *, config, seed):
        super().__init__(name="dsoftki", config=config, seed=seed)
