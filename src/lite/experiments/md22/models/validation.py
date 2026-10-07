from lite.experiments.md22.metrics import normalized_energy_rmse_per_atom, raw_energy_rmse_per_atom


class MD22Validation:
    validation_enabled = False

    def fit(self, split):
        def evaluate(mean):
            return {
                "normalized_energy_rmse_per_atom": normalized_energy_rmse_per_atom(
                    mean, split.y_test, split.n_atoms
                ),
                "raw_energy_rmse_per_atom": raw_energy_rmse_per_atom(
                    mean,
                    split.E_test,
                    energy_mean=split.scaler.energy_mean,
                    energy_std=split.scaler.energy_std,
                    n_atoms=split.n_atoms,
                ),
            }

        self.validation_callback = evaluate if self.validation_enabled else None
        return super().fit(split)
