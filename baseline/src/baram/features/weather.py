import numpy as np
from ..constants import TIME_COL

LDAPS_WIND = {
 "ws10": ("heightAboveGround_10_10u","heightAboveGround_10_10v"),
 "ws50_maxcomp": ("heightAboveGround_50_50MUmax","heightAboveGround_50_50MVmax"),
 "ws50_mincomp": ("heightAboveGround_50_50MUmin","heightAboveGround_50_50MVmin"),
}
GFS_WIND = {
 "ws10":("heightAboveGround_10_10u","heightAboveGround_10_10v"), "ws80":("heightAboveGround_80_u","heightAboveGround_80_v"),
 "ws100":("heightAboveGround_100_100u","heightAboveGround_100_100v"), "ws_pbl":("planetaryBoundaryLayer_0_u","planetaryBoundaryLayer_0_v"),
 "ws850":("isobaricInhPa_850_u","isobaricInhPa_850_v"), "ws700":("isobaricInhPa_700_u","isobaricInhPa_700_v"),
 "ws500":("isobaricInhPa_500_u","isobaricInhPa_500_v")}
THERMO = {"ldaps":["heightAboveGround_2_t","heightAboveGround_2_dpt","heightAboveGround_2_r","surface_0_sp","meanSea_0_prmsl"],
          "gfs":["heightAboveGround_2_2t","heightAboveGround_2_2d","heightAboveGround_2_2r","surface_0_sp","meanSea_0_prmsl"]}

def add_derived(df, kind):
    d=df.copy(); pairs=LDAPS_WIND if kind=="ldaps" else GFS_WIND
    for name,(u,v) in pairs.items(): d[name]=np.hypot(d[u],d[v])
    if kind=="ldaps":
        um=(d["heightAboveGround_50_50MUmax"]+d["heightAboveGround_50_50MUmin"])/2
        vm=(d["heightAboveGround_50_50MVmax"]+d["heightAboveGround_50_50MVmin"])/2
        d["ws50_mid"]=np.hypot(um,vm)
    return d

def summary_features(df, kind, thermodynamic=True):
    d=add_derived(df,kind); wind=list((LDAPS_WIND if kind=="ldaps" else GFS_WIND))
    if kind=="ldaps": wind += ["ws50_mid"]
    if kind=="gfs": wind += ["surface_0_gust"]
    cols=wind + (THERMO[kind] if thermodynamic else [])
    out=d.groupby(TIME_COL)[cols].agg(["mean","max","min","std"])
    out.columns=[f"{kind}_{a}_{b}" for a,b in out.columns]
    return out.reset_index()

