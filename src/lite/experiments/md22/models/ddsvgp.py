from lite.experiments.md22.models.derivative_baselines import DerivativeBaselineModel


class DDSVGPModel(DerivativeBaselineModel):
    def __init__(self, *, config, seed):
        super().__init__(name="ddsvgp", config=config, seed=seed)
