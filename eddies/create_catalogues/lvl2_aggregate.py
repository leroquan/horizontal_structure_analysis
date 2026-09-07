# %%
import os
import dask.bag as db
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


lake = 'geneva'
folder_path = rf"/storage/alplakes_test/{lake}_100m_2025/outputs_swirl"
input_folder = os.path.join(folder_path, "eddy_catalogues_final")

output_folder = os.path.join(folder_path, "eddy_catalogues_final")
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "lvl2.csv")


dist_threshold = 10 # number of cells
time_threshold = 2 # number of timestep
timestep_in_seconds = 3600

# # Import catalogue lvl1

level1_csv_path = os.path.join(input_folder, "lvl1.csv")
level1_data = pd.read_csv(level1_csv_path)


level1_data = level1_data.copy()
level1_data['parsed_date'] = pd.to_datetime(level1_data['date'])
level1_data['time_index'] = level1_data['time_index'].round(0).astype(int)

# # Create catalogue lvl2

def track_eddies_level2(
    level1_data: pd.DataFrame,
    dist_threshold: float,
    time_threshold: int,
):
    df = level1_data.sort_values("time_index").copy()
    df["id_next"] = None

    groups = dict(tuple(df.groupby("time_index")))
    dist2_thr = dist_threshold ** 2

    for t, df_t in groups.items():

        ids_t = df_t["id"].values
        x_t = df_t["xc_mean"].values
        y_t = df_t["yc_mean"].values
        rot_t = df_t["rotation_direction"].values
        dmin_t = df_t["depth_min_[m]"].values
        dmax_t = df_t["depth_max_[m]"].values

        for dt in range(1, time_threshold + 1):
            if t + dt not in groups:
                continue

            df_n = groups[t + dt]

            dx = df_n["xc_mean"].values[:, None] - x_t
            dy = df_n["yc_mean"].values[:, None] - y_t
            dist2 = dx * dx + dy * dy

            dist_ok = dist2 < dist2_thr

            depth_ok = (
                (df_n["depth_min_[m]"].values[:, None] <= dmax_t)
                &
                (df_n["depth_max_[m]"].values[:, None] >= dmin_t)
            )

            rot_ok = (
                df_n["rotation_direction"].values[:, None] == rot_t
            )

            valid = dist_ok & depth_ok & rot_ok

            for j, ref_id in enumerate(ids_t):
                if df.loc[df["id"] == ref_id, "id_next"].notna().any():
                    continue

                matches = np.where(valid[:, j])[0]
                if matches.size > 0:
                    df.loc[df["id"] == ref_id, "id_next"] = (
                        df_n.iloc[matches[0]]["id"]
                    )

    return df


def build_trajectories(df: pd.DataFrame):
    next_map = (
        df[["id", "id_next"]]
        .dropna()
        .set_index("id")["id_next"]
        .to_dict()
    )

    pointed_to = set(next_map.values())
    starts = [i for i in next_map.keys() if i not in pointed_to]

    trajectories = []
    visited = []

    for start in starts:
        traj = [start]
        current = start
        visited.append(start)

        while current in next_map:
            nxt = next_map[current]
            if nxt in visited:
                break
            traj.append(nxt)
            visited.append(nxt)
            current = nxt

        trajectories.append(traj)

    return trajectories



def build_level2_dataframe(level1_data: pd.DataFrame, trajectories):
    rows = []

    for i, traj in enumerate(trajectories):

        aggregated_data = (
            level1_data
            .loc[level1_data["id"].isin(traj)]
            .sort_values("time_index")
        )

        lifespan = (
            aggregated_data["time_index"].iloc[-1]
            - aggregated_data["time_index"].iloc[0]
        )

        rows.append({
            "id": i,
            "id_lvl1": aggregated_data["id"].tolist(),
            "time_indices(t)": aggregated_data["time_index"].tolist(),
            "dates(t)": aggregated_data["date"].tolist(),
            "xc(t)": aggregated_data["xc_mean"].tolist(),
            "yc(t)": aggregated_data["yc_mean"].tolist(),
            "depth_min(t)_[m]": aggregated_data["depth_min_[m]"].tolist(),
            "depth_max(t)_[m]": aggregated_data["depth_max_[m]"].tolist(),
            "volume(t)_[m3]": aggregated_data["volume_[m3]"].tolist(),
            "rotation_direction": aggregated_data.iloc[0]["rotation_direction"],
            "kinetic_energy(t)_[MJ]":
                aggregated_data["kinetic_energy_eddy_[MJ]"].tolist(),
            "lifespan_[h]": lifespan,
        })

    return pd.DataFrame(rows)



tracked = track_eddies_level2(
    level1_data,
    dist_threshold,
    time_threshold
)

trajectories = build_trajectories(tracked)

eddy_rows_lvl2 = build_level2_dataframe(
    tracked,
    trajectories
)


# # Save
eddy_rows_lvl2.to_csv(output_path, index=False)





