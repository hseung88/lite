from lite.methods.names import normalize_method_name

CONFIG_ALIASES = {
    "exact_learn_lengthscale": "standard_gp_learn_lengthscale",
    "exact_learn_outputscale": "standard_gp_learn_outputscale",
    "exact_learn_sigma_f": "standard_gp_learn_sigma_f",
    "exact_log_every": "standard_gp_log_every",
    "exact_lr": "standard_gp_lr",
    "exact_lr_by_dataset": "standard_gp_lr_by_dataset",
    "exact_lr_default": "standard_gp_lr_default",
    "exact_max_train": "standard_gp_max_train",
    "exact_min_sigma_f": "standard_gp_min_sigma_f",
    "exact_train_epochs": "standard_gp_train_epochs",
    "exact_train_steps": "standard_gp_train_steps",
    "exact_weight_decay": "standard_gp_weight_decay",
    "exact_dense_max_d": "standard_dgp_dense_max_d",
    "exact_dense_max_obs_dim": "standard_dgp_dense_max_obs_dim",
    "exact_deroos_max_n": "standard_dgp_deroos_max_n",
    "lite_batch_size": "lite_train_batch_size",
    "tera_batch_size": "tera_train_batch_size",
}


def normalize_config(raw: dict) -> dict:
    result = dict(raw)
    for old, new in CONFIG_ALIASES.items():
        if old not in result:
            continue
        value = result.pop(old)
        if new in result and result[new] != value:
            raise ValueError(f"Conflicting configuration keys: {old} and {new}")
        result[new] = value
    if "methods" in result:
        result["methods"] = list(dict.fromkeys(map(normalize_method_name, result["methods"])))
    return result
