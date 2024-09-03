import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def calculate_pelvis_acs(
    l_asis: np.ndarray, r_asis: np.ndarray, l_psis: np.ndarray, r_psis: np.ndarray
) -> np.ndarray:
    mid_asis = (l_asis + r_asis) / 2
    mid_psis = (l_psis + r_psis) / 2

    x_init = mid_asis - mid_psis  # forward direction is positive
    y = l_asis - mid_asis  # left direction is positive
    z = np.cross(x_init, y)  # upward direction is positive
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    z /= np.linalg.norm(z)
    
    # Create the rotation matrix
    R = np.array([x, y, z]).T

    # Create the homogeneous transformation matrix
    T = np.array([
        [R[0, 0], R[0, 1], R[0, 2], mid_asis[0]],
        [R[1, 0], R[1, 1], R[1, 2], mid_asis[1]],
        [R[2, 0], R[2, 1], R[2, 2], mid_asis[2]],
        [0, 0, 0, 1]
    ])
    return T

def estimate_hip_joint_center(
    l_asis: np.ndarray, r_asis: np.ndarray, l_psis: np.ndarray, r_psis: np.ndarray
) -> dict[str, np.ndarray]:
    # Calculate the pelvic depth and width
    mid_asis = (l_asis + r_asis) / 2
    mid_psis = (l_psis + r_psis) / 2
    pelvic_depth = np.linalg.norm(mid_asis - mid_psis)
    pelvic_width = np.linalg.norm(l_asis - r_asis)

    # Calculate the pelvis anatomical coordinate system, expressed in the global coordinate system
    pelvis_acs = calculate_pelvis_acs(l_asis, r_asis, l_psis, r_psis)

    # Estimate the hip joint centers
    r_hjc = np.array([
        -0.24 * pelvic_depth - 9.9,
        -0.33 * pelvic_width + 7.3,
        -0.30 * pelvic_width - 10.9
    ])
    r_hjc = pelvis_acs @ np.append(r_hjc, 1)

    l_hjc = np.array([
        -0.24 * pelvic_depth - 9.9,
        0.33 * pelvic_width - 7.3,
        -0.30 * pelvic_width - 10.9
    ])
    l_hjc = pelvis_acs @ np.append(l_hjc, 1)
    return {"r_hjc": r_hjc, "l_hjc": l_hjc}

def load_data(filepath: str | Path) -> pd.DataFrame:

    # Parse the filename
    filepath = Path(filepath) if isinstance(filepath, str) else filepath

    # Get the associated channels information
    channels_df = pd.read_csv(
        filepath.parent / filepath.name.replace("_motion.tsv", "_channels.tsv"),
        sep="\t", header=0
    )

    # Get the marker data
    markers_df = pd.read_csv(
        filepath, sep="\t", header=0
    )
    markers_df = markers_df.loc[:, [col for col in markers_df.columns if not col.endswith("_n/a")]]

    # Get the sampling frequency
    sampling_freq_Hz = channels_df["sampling_frequency"].iloc[0].astype(float)
    markers_df.index = np.arange(len(markers_df)) / sampling_freq_Hz
    return markers_df


def main() -> None:
    DATASET_PATH = Path("./datasets/keepcontrol")
    SUB_ID = "pp002"
    TASK_FILE_NAME = "walkPreferred"
    CALIBRATION_FILE_NAME = "calibration2"

    # Load the data
    markers_df = load_data(DATASET_PATH / f"sub-{SUB_ID}" / "motion" / f"sub-{SUB_ID}_task-{TASK_FILE_NAME}_tracksys-omc_motion.tsv")
    static_markers_df = load_data(DATASET_PATH / f"sub-{SUB_ID}" / "motion" / f"sub-{SUB_ID}_task-{CALIBRATION_FILE_NAME}_tracksys-omc_motion.tsv")
    
    # Take the mean position over the static trial, excluding NaN values
    static_markers_df = static_markers_df.mean(axis=0, skipna=True)

    # Extract the marker names
    static_marker_names = [
        mrk[:-6] for mrk in static_markers_df.index if mrk.endswith("_POS_x")
    ]

    # Estimate the hip joint center
    hjc = estimate_hip_joint_center(
        l_asis=static_markers_df[["l_asis_POS_x", "l_asis_POS_y", "l_asis_POS_z"]].values,
        r_asis=static_markers_df[["r_asis_POS_x", "r_asis_POS_y", "r_asis_POS_z"]].values,
        l_psis=static_markers_df[["l_psis_POS_x", "l_psis_POS_y", "l_psis_POS_z"]].values,
        r_psis=static_markers_df[["r_psis_POS_x", "r_psis_POS_y", "r_psis_POS_z"]].values
    )

    # Plot the data
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    for mrk in static_marker_names:
        ax.plot(
            static_markers_df[f"{mrk}_POS_x"],
            static_markers_df[f"{mrk}_POS_y"],
            static_markers_df[f"{mrk}_POS_z"],
            ls="none", marker="o", label=mrk
        )
    for lr in "lr":
        ax.plot(
            (static_markers_df[f"{lr}_ank_POS_x"] + static_markers_df[f"{lr}_st_mm_POS_x"]) / 2,
            (static_markers_df[f"{lr}_ank_POS_y"] + static_markers_df[f"{lr}_st_mm_POS_y"]) / 2,
            (static_markers_df[f"{lr}_ank_POS_z"] + static_markers_df[f"{lr}_st_mm_POS_z"]) / 2,
            ls="none", marker="x", label=f"{lr} ankle joint"
        )
        ax.plot(
            (static_markers_df[f"{lr}_st_fem_POS_x"] + static_markers_df[f"{lr}_st_fel_POS_x"]) / 2,
            (static_markers_df[f"{lr}_st_fem_POS_y"] + static_markers_df[f"{lr}_st_fel_POS_y"]) / 2,
            (static_markers_df[f"{lr}_st_fem_POS_z"] + static_markers_df[f"{lr}_st_fel_POS_z"]) / 2,
            ls="none", marker="x", label=f"{lr} knee joint"
        )
        ax.plot(
            hjc[f"{lr}_hjc"][0], hjc[f"{lr}_hjc"][1], hjc[f"{lr}_hjc"][2],
            ls="none", marker="^", label="Pelvic depth"
        )
    ax.set_aspect("equal")
    plt.show()
    return


if __name__ == "__main__":
    main()