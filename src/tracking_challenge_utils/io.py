from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import numpy as np
import tracksdata as td
import zarr


@dataclass
class Dataset:
    path: Path
    image: da.Array
    tracks: td.graph.IndexedRXGraph
    scale: tuple[float, float, float]

    def napari_tracks(self) -> tuple[np.ndarray, dict[int, int]]:
        return td.functional.to_napari_format(
            self.tracks,
            self.image.shape,
            solution_key=None,
            output_track_id_key="track_id",
            mask_key=None,
        )


def open_dataset(ds_path: Path) -> Dataset:
    image_path = ds_path.parent / f"{ds_path.stem}.zarr"
    tracks_path = ds_path.parent / f"{ds_path.stem}.geff"

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not tracks_path.exists():
        raise FileNotFoundError(f"Tracks file not found: {tracks_path}")

    img_ds = zarr.open_group(image_path, mode="r")
    tracks = td.graph.IndexedRXGraph.from_geff(tracks_path)

    da_arr = da.from_zarr(img_ds["0"]).squeeze(1)

    transform = img_ds.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]

    if transform["type"] != "scale":
        raise ValueError(f"Transform type is not 'scale': {transform}")

    return Dataset(path=ds_path, image=da_arr, tracks=tracks, scale=transform["scale"][-3:])
