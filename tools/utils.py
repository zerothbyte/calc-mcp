import json
import os
from datetime import datetime

import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


SAFE_SYMPY_NAMES = {
    "pi": sympy.pi,
    "e": sympy.E,
    "oo": sympy.oo,
    "inf": sympy.oo,
    "I": sympy.I,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "asin": sympy.asin,
    "acos": sympy.acos,
    "atan": sympy.atan,
    "exp": sympy.exp,
    "log": sympy.log,
    "ln": sympy.ln,
    "sqrt": sympy.sqrt,
    "abs": sympy.Abs,
    "Abs": sympy.Abs,
}

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def parse_json_array(s: str) -> list:
    data = json.loads(s)
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array")
    return data


def parse_json_matrix(s: str) -> list:
    data = json.loads(s)
    if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
        raise ValueError("Input must be a JSON 2D array (list of lists)")
    if len(data) > 100 or any(len(row) > 100 for row in data):
        raise ValueError("Matrix too large (max 100x100)")
    return data


def safe_parse_expr(expression: str, local_vars: dict = None):
    locals_dict = dict(SAFE_SYMPY_NAMES)
    if local_vars:
        locals_dict.update(local_vars)
    return parse_expr(expression, local_dict=locals_dict, transformations=TRANSFORMATIONS)


def expr_to_callable(expression: str, variable: str = "x"):
    var = sympy.Symbol(variable)
    local_vars = {variable: var}
    expr = safe_parse_expr(expression, local_vars)
    return sympy.lambdify(var, expr, modules=["numpy"])


def get_output_path(prefix: str = "plot", extension: str = "png", save_path: str = "") -> str:
    if save_path:
        return save_path
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(OUTPUT_DIR, f"{prefix}_{timestamp}.{extension}")
