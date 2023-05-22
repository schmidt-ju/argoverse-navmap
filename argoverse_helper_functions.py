# This code is from: https://github.com/argoverse/argoverse-api/blob/f886ac54fba9f06f8a7d109eb663c7f501b3aa8e/argoverse/utils/manhattan_search.py
# Obtained on 24.02.2023.
# argoverse-api is licensed under MIT.
# The full license is available in: LICENSE_ARGOVERSE

import numpy as np


def find_all_polygon_bboxes_overlapping_query_bbox(
    polygon_bboxes: np.ndarray, query_bbox: np.ndarray
) -> np.ndarray:
    """Find all the overlapping polygon bounding boxes.
    Each bounding box has the following structure:
        bbox = np.array([x_min,y_min,x_max,y_max])
    In 3D space, if the coordinates are equal (polygon bboxes touch), then these are considered overlapping.
    We have a guarantee that the cropped image will have any sort of overlap with the zero'th object bounding box
    inside of the image e.g. along the x-dimension, either the left or right side of the bounding box lies between the
    edges of the query bounding box, or the bounding box completely engulfs the query bounding box.
    Args:
        polygon_bboxes: An array of shape (K,), each array element is a NumPy array of shape (4,) representing
                        the bounding box for a polygon or point cloud.
        query_bbox: An array of shape (4,) representing a 2d axis-aligned bounding box, with order
                    [min_x,min_y,max_x,max_y].
    Returns:
        An integer array of shape (K,) representing indices where overlap occurs.
    """
    query_min_x = query_bbox[0]
    query_min_y = query_bbox[1]

    query_max_x = query_bbox[2]
    query_max_y = query_bbox[3]

    bboxes_x1 = polygon_bboxes[:, 0]
    bboxes_x2 = polygon_bboxes[:, 2]

    bboxes_y1 = polygon_bboxes[:, 1]
    bboxes_y2 = polygon_bboxes[:, 3]

    # check if falls within range
    overlaps_left = (query_min_x <= bboxes_x2) & (bboxes_x2 <= query_max_x)
    overlaps_right = (query_min_x <= bboxes_x1) & (bboxes_x1 <= query_max_x)

    x_check1 = bboxes_x1 <= query_min_x
    x_check2 = query_min_x <= query_max_x
    x_check3 = query_max_x <= bboxes_x2
    x_subsumed = x_check1 & x_check2 & x_check3

    x_in_range = overlaps_left | overlaps_right | x_subsumed

    overlaps_below = (query_min_y <= bboxes_y2) & (bboxes_y2 <= query_max_y)
    overlaps_above = (query_min_y <= bboxes_y1) & (bboxes_y1 <= query_max_y)

    y_check1 = bboxes_y1 <= query_min_y
    y_check2 = query_min_y <= query_max_y
    y_check3 = query_max_y <= bboxes_y2
    y_subsumed = y_check1 & y_check2 & y_check3
    y_in_range = overlaps_below | overlaps_above | y_subsumed

    overlap_indxs = np.where(x_in_range & y_in_range)[0]
    return overlap_indxs
