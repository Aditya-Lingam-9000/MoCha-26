from __future__ import annotations

import ast
import re
from argparse import Namespace


def _is_float(value: str) -> bool:
    try:
        return re.match(r"^[-+]?[0-9]+\.[0-9]+$", str(value).strip()) is not None
    except Exception:
        return False


def _is_int(value: str) -> bool:
    value = str(value).strip().lstrip("-").lstrip("+")
    return value.isdigit()


def get_opt(opt_path, device, **kwargs):
    opt = Namespace()
    opt_dict = vars(opt)
    skip = {"-------------- End ----------------", "------------ Options -------------", ""}

    print("Reading", opt_path)
    with open(opt_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip("\n")
            if line.strip() in skip:
                continue
            key, value = line.split(": ", 1)
            if value in {"True", "False"}:
                opt_dict[key] = value == "True"
            elif _is_float(value):
                opt_dict[key] = float(value)
            elif _is_int(value):
                opt_dict[key] = int(value)
            else:
                try:
                    opt_dict[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    opt_dict[key] = str(value)

    opt.which_epoch = "finest"
    opt.device = device
    opt_dict.update(kwargs)
    return opt
