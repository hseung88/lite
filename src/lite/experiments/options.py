"""Shared method options and explicit configuration precedence."""

from dataclasses import fields, replace

LOCAL_MD22 = ("standard_gp", "lite", "tera", "vecchia")
MINIBATCH_MD22 = ("lite", "tera", "vecchia", "dsoftki", "ddsvgp")
DERIVATIVE = ("lite", "tera")
METHOD_PREFIXES = {
    "bo": ("lite", "tera", "vbo", "turbo"),
    "md22": (*LOCAL_MD22, "dsoftki", "ddsvgp"),
}
METHOD_ALIASES = {
    "tera_batched": "tera",
    "tera-target": "tera",
    "tera-target-pred": "tera",
    "turbo-logei": "turbo",
    "lite_normalized": "lite",
}


def option_targets(task):
    """Map common names to the concrete dataclass fields they control."""
    groups = {
        "bo": {
            **dict.fromkeys(
                (
                    "m",
                    "lr",
                    "train_steps",
                    "train_batch_size",
                    "prediction_batch_size",
                    "initial_train_steps",
                    "update_train_steps",
                    "refit_every",
                    "lengthscale_init",
                    "base_lengthscale",
                    "lengthscale_min",
                    "lengthscale_max",
                    "learn_lengthscale",
                    "learn_outputscale",
                    "learn_sigma_f",
                    "learn_sigma_g",
                    "sigma_f",
                    "sigma_g",
                    "gradient_noise_model",
                    "acq_raw_samples",
                    "acq_restarts",
                    "acq_maxiter",
                ),
                DERIVATIVE,
            ),
        },
        "md22": {
            "m": ("lite", "tera", "vecchia"),
            "lr": (*LOCAL_MD22, "dsoftki", "ddsvgp"),
            "train_steps": LOCAL_MD22,
            "train_epochs": (*LOCAL_MD22, "dsoftki", "ddsvgp"),
            "train_batch_size": MINIBATCH_MD22,
            "prediction_batch_size": MINIBATCH_MD22,
            "weight_decay": LOCAL_MD22,
            "learn_lengthscale": LOCAL_MD22,
            "learn_outputscale": LOCAL_MD22,
            "learn_sigma_f": LOCAL_MD22,
            "learn_sigma_g": DERIVATIVE,
            "gradient_noise_model": DERIVATIVE,
            "graph_refresh_epochs": ("lite", "tera", "vecchia"),
            "log_every": LOCAL_MD22,
        },
    }[task]
    result = {}
    for name, prefixes in groups.items():
        result[name] = {
            prefix: f"{prefix}_{'batch_size' if name == 'train_batch_size' and prefix in {'dsoftki', 'ddsvgp'} else name}"
            for prefix in prefixes
        }
    return result


def method_prefix(method):
    name = method.strip().lower().replace(" ", "_")
    return METHOD_ALIASES.get(name, name)


def _expand(values, task):
    targets = option_targets(task)
    result = {}
    for name, value in values.items():
        if name in targets:
            result.update(dict.fromkeys(targets[name].values(), value))
    result.update({name: value for name, value in values.items() if name not in targets})
    if task == "md22" and "m" in values:
        result["m"] = values["m"]
    # A step shorthand controls both BO stages unless specified in this layer.
    if task == "bo":
        for prefix in DERIVATIVE:
            key = f"{prefix}_train_steps"
            if key in result:
                for stage in ("initial", "update"):
                    result.setdefault(f"{prefix}_{stage}_train_steps", result[key])
    return result


def expand_yaml(raw, task, config_type):
    """Method YAML > shared YAML; legacy flat method fields remain valid."""
    raw = dict(raw)
    shared = raw.pop("shared", {})
    methods = raw.pop("method_options", {})
    if not isinstance(shared, dict) or not isinstance(methods, dict):
        raise ValueError("shared and method_options must be mappings.")
    field_names = {f.name for f in fields(config_type)}
    common_names = set(option_targets(task))
    allowed_shared = common_names | (
        field_names
        - {
            name
            for name in field_names
            if any(name.startswith(prefix + "_") for prefix in METHOD_PREFIXES[task])
        }
    )
    unknown = shared.keys() - allowed_shared
    if unknown:
        raise ValueError(f"Unknown shared option(s): {sorted(unknown)}")
    flat = dict(raw)
    nested = {}
    for method, options in methods.items():
        prefix = method_prefix(method)
        if prefix not in METHOD_PREFIXES[task] or not isinstance(options, dict):
            raise ValueError(f"Invalid method_options entry: {method!r}")
        for name, value in options.items():
            key = option_targets(task).get(name, {}).get(prefix, f"{prefix}_{name}")
            if key not in field_names:
                raise ValueError(f"Unsupported {method} option: {name!r}")
            if key in flat and flat[key] != value:
                raise ValueError(f"Conflicting flat and nested option: {key}")
            if key in nested and nested[key] != value:
                raise ValueError(f"Conflicting method aliases for option: {key}")
            nested[key] = value
    result = _expand(shared, task)
    # Preserve the meaning of native fields in existing flat YAML files.
    native = {
        name: flat.pop(name) for name in list(flat) if name in field_names and name in common_names
    }
    result.update(_expand({**flat, **nested}, task))
    result.update(native)
    if task == "md22" and "m" in native:
        for key in option_targets(task)["m"].values():
            if key not in flat and key not in nested:
                result[key] = None
    unknown = result.keys() - field_names
    if unknown:
        raise ValueError(f"Unknown configuration option(s): {sorted(unknown)}")
    if task == "md22":
        for prefix in LOCAL_MD22:
            steps, epochs = f"{prefix}_train_steps", f"{prefix}_train_epochs"
            if (
                "train_steps" in shared
                and epochs not in flat
                and epochs not in nested
                and "train_epochs" not in shared
            ):
                result[epochs] = 0
            if steps in nested and epochs not in nested and epochs not in flat:
                result[epochs] = 0
    else:
        result.update(
            {
                name: value
                for name, value in shared.items()
                if name in field_names and name in common_names and name not in native
            }
        )
    return result


def add_shared_arguments(parser, config_type, task):
    from lite.experiments.cli import add_typed_argument

    names = {f.name for f in fields(config_type)}
    for name, targets in option_targets(task).items():
        flag = "--" + name.replace("_", "-")
        scopes = ", ".join(targets)
        help_text = (
            f"Shared {name.replace('_', ' ')} for {scopes}; method-specific flags override it."
        )
        if name in names:
            help_text += " Also sets the existing global configuration field."
        if flag in parser._option_string_actions:
            parser._option_string_actions[flag].help = help_text
            continue
        field = next(field for field in targets.values() if field in names)
        add_typed_argument(
            parser,
            config_type,
            field,
            flag=flag,
            dest=name,
            help=help_text,
        )


def resolve_cli(cfg, args, task):
    """Method CLI > common CLI > the already resolved YAML configuration."""
    values = vars(args)
    targets = option_targets(task)
    field_names = {f.name for f in fields(cfg)}
    selected = {method_prefix(m) for m in cfg.methods}
    common = {}
    for name, mapping in targets.items():
        value = values.get(name)
        if value is None:
            continue
        applicable = selected & mapping.keys()
        # MD22's original --m is harmless for nonlocal baselines.
        if not applicable and name not in field_names:
            raise ValueError(f"--{name.replace('_', '-')} does not apply to methods {cfg.methods}.")
        common.update({mapping[p]: value for p in applicable})
    ordinary = {
        name: value
        for name, value in values.items()
        if name in field_names
        and name not in targets
        and name not in {"methods", "datasets", "benchmarks", "seeds"}
        and value is not None
    }
    native = {
        name: values[name]
        for name in targets
        if name in field_names and values.get(name) is not None
    }
    common = _expand(common, task)
    ordinary = _expand(ordinary, task)
    ordinary.update(native)
    if task == "md22":
        for prefix in LOCAL_MD22:
            steps, epochs = f"{prefix}_train_steps", f"{prefix}_train_epochs"
            if steps in common and epochs not in common:
                common[epochs] = 0
            if steps in ordinary and epochs not in ordinary:
                ordinary[epochs] = 0
    return replace(cfg, **{**common, **ordinary})


def method_snapshot(cfg, method):
    """Record configured method options after all CLI/YAML overrides."""
    from dataclasses import asdict

    prefix = method_prefix(method)
    return {
        name[len(prefix) + 1 :]: value
        for name, value in asdict(cfg).items()
        if name.startswith(prefix + "_")
    }
