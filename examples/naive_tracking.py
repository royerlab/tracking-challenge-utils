from pathlib import Path

import dask.array as da
import napari
import numpy as np
import scipy.ndimage as ndi
import tracksdata as td
from skimage.feature import peak_local_max
from tqdm import tqdm

from tracking_challenge_utils.io import open_dataset
from tracking_challenge_utils.metrics import evaluate


def naive_cell_detection(
    image: da.Array,
    scale: tuple[float, float, float],
    sigma: float = 1,
    threshold_abs: float = 200,
) -> np.ndarray:
    """
    Naive cell detection using peak local max.

    Parameters
    ----------
    image : dask.array.Array
        The image to detect cells in.
    scale : tuple[float, float, float]
        The scale of the image.
    sigma : float, optional
        The sigma of the Gaussian filter
    threshold_abs : float, optional
        The absolute threshold for peak local max

    Returns
    -------
    np.ndarray
        (T, Z, Y, X) array of detected cells centers.
    """
    sigma = sigma / np.array(scale)
    coords = []
    for t in tqdm(range(image.shape[0]), desc="Finding centroids"):
        # loading dask array as numpy array to avoid computational graph
        frame = np.asarray(image[t])
        frame = ndi.gaussian_filter(frame, sigma=sigma)
        pos = peak_local_max(frame, threshold_abs=threshold_abs)
        pos = np.concatenate([np.full((pos.shape[0], 1), t), pos], axis=1)
        coords.append(pos)

    return np.concatenate(coords, axis=0, dtype=float)


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

    coords = naive_cell_detection(ds.image, ds.scale)

    # empty graph to store candidate cells
    candidate_graph = td.graph.InMemoryGraph()

    # adding new keys, "t" exists by default
    for c in ["z", "y", "x"]:
        candidate_graph.add_node_attr_key(c, default_value=-1)

    # adding nodes to the graph
    candidate_graph.bulk_add_nodes([{"t": t, "z": z, "y": y, "x": x} for t, z, y, x in coords.tolist()])

    # adding candidate cells to the viewer
    viewer.add_points(
        coords, ndim=4, size=10, opacity=1.0, face_color="transparent", border_color="red", name="Candidate cells"
    )

    # connected nodes between adjacent frames using distance
    # this could be improved by using operating in physical units space
    td.edges.DistanceEdges(distance_threshold=5, n_neighbors=5, output_key="distance").add_edges(candidate_graph)

    # finds solution that minimizes the sum of edge weights
    solution_graph = td.solvers.NearestNeighborsSolver(
        edge_weight=-td.EdgeAttr("distance"),
        max_children=1,  # no divisions because detection is not that great
    ).solve(candidate_graph)

    # exporting solution to napari format
    solution_tracklets, solution_tracklets_graph = td.functional.to_napari_format(
        graph=solution_graph,
        shape=ds.image.shape,
    )

    # adding solution tracks to the viewer
    viewer.add_tracks(
        solution_tracklets,
        graph=solution_tracklets_graph,
        tail_width=5,
        tail_length=100,
        opacity=1.0,
        name="Solution tracks",
        scale=ds.scale,
    )

    # masks are required for metrics computation
    td.nodes.MaskDiskAttrs(radius=10, image_shape=ds.image.shape[1:]).add_node_attrs(solution_graph)

    td.nodes.MaskDiskAttrs(radius=10, image_shape=ds.image.shape[1:]).add_node_attrs(ds.tracks)

    # computing metrics
    metrics = evaluate(solution_graph, ds.tracks, metric="jaccard")
    print(f"Final score: {metrics:.4f}")

    napari.run()


if __name__ == "__main__":
    main()
