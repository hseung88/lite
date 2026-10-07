def resolve_batch_size(value, legacy, *, default, name):
    if legacy is not None:
        if value is not None and int(value) != int(legacy):
            raise ValueError(f"Conflicting {name} and legacy batch_size")
        value = legacy
    size = default if value is None else int(value)
    if size <= 0:
        raise ValueError(f"{name} must be positive")
    return size
