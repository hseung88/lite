METHOD_ALIASES = {
    "Exact GP": "Standard GP",
    "Exact dGP": "Standard dGP",
    "Exact dGP-deRoos": "Standard dGP-deRoos",
    "LITE_normalized": "lite_normalized",
}


def normalize_method_name(name: str) -> str:
    return METHOD_ALIASES.get(name, name)


def normalize_md22_method_name(name: str) -> str:
    """Canonical method names for CLI arguments, configurations and results."""
    key = name.strip().lower().replace(" ", "_")
    key = {"exact_gp": "standard_gp", "lite_normalized": "lite", "vecchia_gp": "vecchia"}.get(
        key, key
    )
    if key not in {"standard_gp", "dsoftki", "ddsvgp", "tera", "tera_batched", "lite", "vecchia"}:
        raise ValueError(f"Unknown MD22 method: {name}")
    return key
