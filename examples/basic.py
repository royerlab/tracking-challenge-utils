from pathlib import Path

import napari

from tracking_challenge_utils.io import open_dataset


def main() -> None:
    """
    Basic example of how to visualize a dataset in napari.
    """

    DATA_DIR = Path("/hpc/projects/group.royer/imaging/tracking-challenge/2024_03_22_dorado")
    ds_name = "0001_0190_1651_0467"

    # load dataset object
    ds = open_dataset(DATA_DIR / ds_name)

    viewer = napari.Viewer()

    # add ROI of interest
    viewer.add_image(ds.image, scale=ds.scale, contrast_limits=(0, 2500))

    # load `ds.tracks` into a napari compatible format
    tracklets, tracklets_graph = ds.napari_tracks()
    viewer.add_tracks(
        tracklets,
        graph=tracklets_graph,
        scale=ds.scale,
        tail_width=5,
        tail_length=100,
        opacity=1.0,
    )

    # load `ds.tracks` into napari as points (no edges)
    coordinates = ds.tracks.node_attrs(attr_keys=["t", "z", "y", "x"]).to_numpy()
    viewer.add_points(
        coordinates, ndim=4, scale=ds.scale, size=10, opacity=1.0, face_color="transparent", border_color="red"
    )

    napari.run()


if __name__ == "__main__":
    main()
