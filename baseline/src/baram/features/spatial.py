import re
import numpy as np
from ..constants import TIME_COL
from .weather import add_derived, LDAPS_WIND, GFS_WIND, THERMO

def _decimal(s):
    nums=[float(x) for x in re.findall(r"\d+(?:\.\d+)?",str(s))]
    if len(nums)<6: raise ValueError(f"Cannot parse coordinate: {s}")
    lat=nums[0]+nums[1]/60+nums[2]/3600; lon=nums[3]+nums[4]/60+nums[5]/3600
    return lat,lon

def group_centres(metadata):
    m=metadata.copy(); m["KPX그룹"]=m["KPX그룹"].ffill().astype(int)
    coords=m["좌표(Google)"].map(_decimal); m["lat"]=[x[0] for x in coords]; m["lon"]=[x[1] for x in coords]
    return m.groupby("KPX그룹")[["lat","lon"]].mean()

def nearest_grid_ids(weather, centres):
    grids=weather.groupby("grid_id")[["latitude","longitude"]].first()
    return {int(g): grids.assign(dist=(grids.latitude-r.lat)**2+(grids.longitude-r.lon)**2).dist.idxmin() for g,r in centres.iterrows()}

def nearest_features(df, kind, centres, thermodynamic=True):
    d=add_derived(df,kind); ids=nearest_grid_ids(d,centres)
    cols=list((LDAPS_WIND if kind=="ldaps" else GFS_WIND)) + (["ws50_mid"] if kind=="ldaps" else ["surface_0_gust"])
    if thermodynamic: cols += THERMO[kind]
    parts=[]
    for group,gid in ids.items():
        p=d.loc[d.grid_id==gid,[TIME_COL,*cols]].copy(); p=p.rename(columns={c:f"{kind}_g{group}_nearest_{c}" for c in cols}); parts.append(p)
    out=parts[0]
    for p in parts[1:]: out=out.merge(p,on=TIME_COL,how="inner",validate="one_to_one")
    return out,ids

