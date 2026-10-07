import argparse
from dataclasses import fields
from functools import lru_cache
from types import UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints


@lru_cache(maxsize=None)
def _type_hints(config_type):
    return get_type_hints(config_type)


def add_typed_argument(parser, config_type, name, *, flag=None, dest=None, **kwargs):
    annotation = _type_hints(config_type)[name]
    options = get_args(annotation)
    if get_origin(annotation) in (UnionType, Union):
        annotation = next(t for t in options if t is not type(None))
    choices = None
    if get_origin(annotation) is Literal:
        choices = get_args(annotation)
        typ = type(choices[0])
    else:
        typ = annotation
    flag = flag or "--" + name.replace("_", "-")
    dest = dest or name
    kwargs.setdefault("default", None)
    if typ is bool:
        parser.add_argument(flag, dest=dest, action=argparse.BooleanOptionalAction, **kwargs)
    elif typ in (str, int, float):
        parser.add_argument(flag, dest=dest, type=typ, choices=choices, **kwargs)


def add_config_arguments(parser, config_type, *, prefixes=(), names=()):
    existing = {a.dest for a in parser._actions}
    for field in fields(config_type):
        name = field.name
        if name in existing or not (name in names or name.startswith(prefixes)):
            continue
        add_typed_argument(parser, config_type, name)


def apply_config_arguments(args, config, *, exclude=()):
    for field in fields(config):
        if field.name in exclude:
            continue
        value = getattr(args, field.name, None)
        if value is not None:
            setattr(config, field.name, value)
