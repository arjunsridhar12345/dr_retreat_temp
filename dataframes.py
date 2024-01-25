from __future__ import annotations

import typing

import npc_lims
import pandas as pd
import utils


def get_dfs(version: str | None = None, with_bool_columns: bool = True) -> typing.Mapping[str | npc_lims.NWBComponentStr, pd.DataFrame]:
    """Get a dictionary of dataframes for each table-like component in an NWB file (except units)."""
    components = (c for c in typing.get_args(npc_lims.NWBComponentStr) if c != "units")
    def _helper(component, version, with_bool_columns) -> pd.DataFrame:
        df = pd.read_parquet(npc_lims.get_cache_path(component, version=version))
        if with_bool_columns:
            df = utils.add_bool_columns(df)
        return df
    return utils.LazyDict({
        component: (_helper, (component, version, with_bool_columns), {})
        for component in components
    })