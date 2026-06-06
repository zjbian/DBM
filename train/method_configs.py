"""DBM-Bid method configuration.

This release ships a single method---DBM-Bid (the paper's full model, internally the
``v2`` backbone with the awr-beta5 training stack). Its fully-resolved configuration is
stored in ``configs/dbm_bid.json``; ``build_method_config`` returns a copy of it for any
requested method name, so existing training scripts work unchanged.
"""
import json
import os
from copy import deepcopy

_CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "dbm_bid.json"
)
with open(_CFG_PATH, "r", encoding="utf-8") as _f:
    _DBM_CONFIG = json.load(_f)

# keep both the public name and the internal name pointing at the same config
METHOD_CONFIGS = {"dbm_bid": _DBM_CONFIG, "msdt_v2_awr_beta5_fixed": _DBM_CONFIG}


def build_method_config(method_name: str) -> dict:
    return deepcopy(_DBM_CONFIG)
