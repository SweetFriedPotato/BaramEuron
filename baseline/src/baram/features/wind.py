import numpy as np


LDAPS_WIND_PAIRS = {
    "ws10": ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
    "ws50_maxcomp": ("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax"),
    "ws50_mincomp": ("heightAboveGround_50_50MUmin", "heightAboveGround_50_50MVmin"),
}

GFS_WIND_PAIRS = {
    "ws10": ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
    "ws80": ("heightAboveGround_80_u", "heightAboveGround_80_v"),
    "ws100": ("heightAboveGround_100_100u", "heightAboveGround_100_100v"),
    "ws_pbl": ("planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v"),
    "ws850": ("isobaricInhPa_850_u", "isobaricInhPa_850_v"),
    "ws700": ("isobaricInhPa_700_u", "isobaricInhPa_700_v"),
    "ws500": ("isobaricInhPa_500_u", "isobaricInhPa_500_v"),
}


def add_wind_features(df, kind):
    out = df.copy()
    pairs = LDAPS_WIND_PAIRS if kind == "ldaps" else GFS_WIND_PAIRS
    for name, (u_col, v_col) in pairs.items():
        out[name] = np.hypot(out[u_col], out[v_col])
    if kind == "ldaps":
        u_mid = (out["heightAboveGround_50_50MUmax"] + out["heightAboveGround_50_50MUmin"]) / 2
        v_mid = (out["heightAboveGround_50_50MVmax"] + out["heightAboveGround_50_50MVmin"]) / 2
        out["ws50_mid"] = np.hypot(u_mid, v_mid)
    else:
        out["gust"] = out["surface_0_gust"]
    return out


def wind_feature_names(kind):
    if kind == "ldaps":
        return ["ws10", "ws50_mid", "ws50_maxcomp", "ws50_mincomp"]
    return ["ws10", "ws80", "ws100", "ws_pbl", "ws850", "ws700", "ws500", "gust"]


def wind_formulas(kind):
    if kind == "ldaps":
        return {
            "ws10": "sqrt(heightAboveGround_10_10u^2 + heightAboveGround_10_10v^2)",
            "ws50_mid": "sqrt(((50MUmax + 50MUmin)/2)^2 + ((50MVmax + 50MVmin)/2)^2)",
            "ws50_maxcomp": "sqrt(50MUmax^2 + 50MVmax^2); component-wise maximum, not actual max wind speed",
            "ws50_mincomp": "sqrt(50MUmin^2 + 50MVmin^2); component-wise minimum, not actual min wind speed",
        }
    return {
        "ws10": "sqrt(heightAboveGround_10_10u^2 + heightAboveGround_10_10v^2)",
        "ws80": "sqrt(heightAboveGround_80_u^2 + heightAboveGround_80_v^2)",
        "ws100": "sqrt(heightAboveGround_100_100u^2 + heightAboveGround_100_100v^2)",
        "ws_pbl": "sqrt(planetaryBoundaryLayer_0_u^2 + planetaryBoundaryLayer_0_v^2)",
        "ws850": "sqrt(isobaricInhPa_850_u^2 + isobaricInhPa_850_v^2)",
        "ws700": "sqrt(isobaricInhPa_700_u^2 + isobaricInhPa_700_v^2)",
        "ws500": "sqrt(isobaricInhPa_500_u^2 + isobaricInhPa_500_v^2)",
        "gust": "surface_0_gust",
    }
