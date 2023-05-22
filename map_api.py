import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely.geometry import LineString

from argoverse_helper_functions import (
    find_all_polygon_bboxes_overlapping_query_bbox,
)

# City IDs from newest to oldest
MIAMI_ID = 10316
PITTSBURGH_ID = 10314

_PathLike = Union[str, "os.PathLike[str]"]

CENTERLINE_SAMPLING_DISTANCE = 2.0


class LaneSegment:
    def __init__(
        self,
        id: int,
        has_traffic_control: bool,
        turn_direction: str,
        is_intersection: bool,
        l_neighbor_id: Optional[int],
        r_neighbor_id: Optional[int],
        predecessors: List[int],
        successors: Optional[List[int]],
        centerline: np.ndarray,
    ) -> None:
        """Lane segment class. In the case of this navigation map api,
            roads instead of lane segments are used. We still keep the
            wording "lane" in order to be transferable to the original
            Argoverse api that is HD map-reliant.

        Args:
            id (int): Lane segment id.
            has_traffic_control (bool): Whether the lane segment has a
                traffic control or not. Currently unused.
            turn_direction (str): Turn direction of the lane
                segment ("LEFT", "RIGHT", "NONE"). Currently unused.
            is_intersection (bool): Whether the lane segment has a
                intesection or not. Currently unused.
            l_neighbor_id (Optional[int]): Left neighbor lane segment id.
                Currently unused.
            r_neighbor_id (Optional[int]): Right neighbor lane segment id.
                Currently unused.
            predecessors (List[int]): Predecessing lane segment ids.
            successors (Optional[List[int]]): Successing lane segment ids.
            centerline (np.ndarray): Centerline of the lane segment.
        """
        self.id = id
        self.has_traffic_control = has_traffic_control
        self.turn_direction = turn_direction
        self.is_intersection = is_intersection
        self.l_neighbor_id = l_neighbor_id
        self.r_neighbor_id = r_neighbor_id
        self.predecessors = predecessors
        self.successors = successors
        self.centerline = centerline


class ArgoverseNavMap:
    def __init__(self, root: _PathLike = None):
        """Api that mimics the basic functions of the original Argoverse map api.

        Args:
            root (_PathLike): Root path that contains the map files.
        """
        if root is not None:
            self.root = root
        else:
            root_folder = os.path.dirname(os.path.abspath(__file__))
            self.root = os.path.join(root_folder, "maps")

        # Similar to original Argoverse map api
        self.city_name_to_city_id_dict = {"PIT": PITTSBURGH_ID, "MIA": MIAMI_ID}

        # Load roadgraphs
        self.graphs = {}
        for city_name in self.city_name_to_city_id_dict:
            self.graphs[city_name] = self.load_graph(
                os.path.join(self.map_files_root, f"{city_name}.json")
            )

        # Dictionary for shapely linestrings
        self.city_to_lines_dict = {}
        # Dictionary to map ids to edge tuples
        self.city_edge_id_to_tuple_dict = {}
        # Dictionary to save road polygons, enabling fast queries
        self.city_halluc_bbox_table = {}

        for city_name in self.city_name_to_city_id_dict:
            # Init sublist or subdictionaries
            if city_name not in self.city_to_lines_dict:
                self.city_to_lines_dict[city_name] = []
                self.city_edge_id_to_tuple_dict[city_name] = {}
                self.city_halluc_bbox_table[city_name] = []
            # Fill dictionaries
            for i, edge in enumerate(self.graphs[city_name].edges):
                s = self.graphs[city_name].nodes[edge[0]]
                t = self.graphs[city_name].nodes[edge[1]]
                self.city_to_lines_dict[city_name].append(
                    LineString([np.array([s["x"], s["y"]]), np.array([t["x"], t["y"]])])
                )

                self.city_edge_id_to_tuple_dict[city_name][i] = edge

                self.city_halluc_bbox_table[city_name].append(
                    np.asarray(
                        [
                            min(s["x"], t["x"]),
                            min(s["y"], t["y"]),
                            max(s["x"], t["x"]),
                            max(s["y"], t["y"]),
                        ]
                    )
                )

        # Dictionary to map edge tuples to ids (inverse lookup)
        self.city_tuple_to_edge_id_dict = {}
        for city_name in self.city_name_to_city_id_dict:
            self.city_tuple_to_edge_id_dict[city_name] = dict(
                zip(
                    self.city_edge_id_to_tuple_dict[city_name].values(),
                    self.city_edge_id_to_tuple_dict[city_name].keys(),
                )
            )

            # List of 1d arrays to 2d array
            self.city_halluc_bbox_table[city_name] = np.stack(
                self.city_halluc_bbox_table[city_name], axis=0
            )

        # Dictionary for the actual LaneSegments
        self.city_lane_centerlines_dict = {}
        for city_name in self.city_name_to_city_id_dict:
            if city_name not in self.city_lane_centerlines_dict:
                self.city_lane_centerlines_dict[city_name] = {}
            for i, edge in enumerate(self.graphs[city_name].edges):
                centerline = self.get_interpolated_edge(
                    i, city_name, distance=CENTERLINE_SAMPLING_DISTANCE
                )
                predecessors = self.get_lane_segment_predecessor_ids(i, city_name)
                successors = self.get_lane_segment_successor_ids(i, city_name)
                segment = LaneSegment(
                    i,
                    None,
                    "NONE",
                    None,
                    None,
                    None,
                    predecessors,
                    successors,
                    centerline,
                )
                self.city_lane_centerlines_dict[city_name][i] = segment

    @property
    def map_files_root(self) -> Path:
        if self.root is None:
            raise ValueError("Map root directory cannot be None!")
        return Path(self.root).resolve()

    def load_graph(self, filename: str) -> nx.classes.digraph.DiGraph:
        """Load networkx graph from file.

        Args:
            filename (str): Path to networkx json file.

        Returns:
            nx.classes.digraph.DiGraph: Loaded networkx graph.
        """
        json_data = json.load(open(filename, "r"))
        G = nx.adjacency_graph(json_data)
        return G

    def get_interpolated_edge(
        self,
        edge_id: int,
        city_name: str,
        distance: float = 1.0,
        include_endpoint: bool = True,
    ) -> np.ndarray:
        """Linear interpolation to resample a graph edge into
            points with equal distance.

        Args:
            edge_id (int): Index of the edge.
            city_name (str): City name.
            distance (float, optional): Resampling distance. Defaults to 1.0.
            include_endpoint (bool, optional): Add endpoint after resampling. Defaults to True.

        Returns:
            np.ndarray: Resampled edge.
        """
        line = self.city_to_lines_dict[city_name][edge_id]
        roadline = self._interpolate_linestring(
            line, distance=distance, include_endpoint=include_endpoint
        )

        return roadline

    def _interpolate_linestring(
        self,
        linestring: LineString,
        distance: float,
        include_endpoint: bool,
    ) -> np.ndarray:
        """Linear interpolation to resample a shapely linestring into
            points with equal distance.

        Args:
            linestring (LineString): Shapely linestring.
            distance (float): Resampling distance.
            include_endpoint (bool): Add endpoint after resampling.

        Returns:
            np.ndarray: Resampled linestring.
        """
        sample_points = np.arange(0, linestring.length, distance)
        points = [
            np.asarray(linestring.interpolate(point).xy) for point in sample_points
        ]
        if include_endpoint:
            points += [np.asarray(linestring.boundary.geoms[1].xy)]

        return np.stack(points, axis=0).squeeze()

    def get_lane_ids_in_xy_bbox(
        self,
        query_x: float,
        query_y: float,
        city_name: str,
        query_search_range_manhattan: float = 5.0,
    ) -> List[int]:
        """Get ids of lane segments that are within a manhattan distance-based
            bounding box. This implementation is vectorized.

        Args:
            query_x (float): x coordinate of the query location.
            query_y (float): y coordinate of the query location.
            city_name (str): City name.
            query_search_range_manhattan (float, optional): Search range. Defaults to 5.0.

        Returns:
            List[int]: List of queried lane segments.
        """
        query_min_x = query_x - query_search_range_manhattan
        query_max_x = query_x + query_search_range_manhattan
        query_min_y = query_y - query_search_range_manhattan
        query_max_y = query_y + query_search_range_manhattan

        overlap_indxs = find_all_polygon_bboxes_overlapping_query_bbox(
            self.city_halluc_bbox_table[city_name],
            np.array([query_min_x, query_min_y, query_max_x, query_max_y]),
        )

        return list(overlap_indxs)

    def get_lane_segment_centerline(
        self, lane_segment_id: int, city_name: str
    ) -> np.ndarray:
        """Get 3d centerline of a lane segment.

        Args:
            lane_segment_id (int): Id of the lane segment.
            city_name (str): City name.

        Returns:
            np.ndarray: 3d centerline of the requested lane segment.
        """
        lane_centerline = self.city_lane_centerlines_dict[city_name][
            lane_segment_id
        ].centerline

        # In the original api, the z coordinate gets queried from the ground height map
        # We do not have ground height and therefore always set z=0
        lane_centerline_3d = np.zeros((lane_centerline.shape[0], 3))
        lane_centerline_3d[:, 0:2] = lane_centerline

        return lane_centerline_3d

    def lane_is_in_intersection(self, lane_segment_id: int, city_name: str) -> bool:
        """Get the intersection flag of a lane segment.
            The values are currently set to None by default.

        Args:
            lane_segment_id (int): Id of the lane segment.
            city_name (str): City name.

        Returns:
            bool: Intersection flag of the requested lane segment.
        """
        return self.city_lane_centerlines_dict[city_name][
            lane_segment_id
        ].is_intersection

    def get_lane_turn_direction(self, lane_segment_id: int, city_name: str) -> str:
        """Get the turn flag of a lane segment.
            The values are currently set to "NONE" by default.

        Args:
            lane_segment_id (int): Id of the lane segment.
            city_name (str): City name.

        Returns:
            str: Turn flag of the requested lane segment ("LEFT", "RIGHT", "NONE").
        """
        return self.city_lane_centerlines_dict[city_name][
            lane_segment_id
        ].turn_direction

    def lane_has_traffic_control_measure(
        self, lane_segment_id: int, city_name: str
    ) -> bool:
        """Get the intersection flag of a lane segment.
            The values are currently set to None by default.

        Args:
            lane_segment_id (int): Id of the lane segment.
            city_name (str): City name.

        Returns:
            bool: Intersection flag of the requested lane segment.
        """
        return self.city_lane_centerlines_dict[city_name][
            lane_segment_id
        ].has_traffic_control

    def get_lane_segment_polygon(
        self, lane_segment_id: int, city_name: str
    ) -> np.ndarray:
        """Get the polygon of a lane segment.
            The values are currently set to a zero array by default.

        Args:
            lane_segment_id (int): Id of the lane segment.
            city_name (str): City name.

        Returns:
            np.ndarray: Polygon of the requested lane segment.
        """
        return np.zeros([2, 3])

    def get_lane_segment_predecessor_ids(
        self, lane_segment_id: int, city_name: str
    ) -> List[int]:
        """Get predecessors of a lane segment. Self-loops are removed.

        Args:
            edge_id (int): Id of the lane segment.
            city_name (str): City name.

        Returns:
            List[int]: Ids of predecessing lane segments.
        """
        edge = self.city_edge_id_to_tuple_dict[city_name][lane_segment_id]
        # Return all nodes that result in the start point of the edge
        preds = list(self.graphs[city_name].in_edges(edge[0]))

        ids = [
            self.city_tuple_to_edge_id_dict[city_name][pred]
            for pred in preds
            if edge[1] != pred[0]
        ]

        return ids

    def get_lane_segment_successor_ids(self, edge_id: int, city_name: str) -> List[int]:
        """Get successors of a lane segment. Self-loops are removed.

        Args:
            edge_id (int): Id of the lane segment.
            city_name (str): City name.

        Returns:
            List[int]: Ids of successing lane segments.
        """
        edge = self.city_edge_id_to_tuple_dict[city_name][edge_id]
        # Return all nodes that result from the endpoint of the edge
        succs = list(self.graphs[city_name].out_edges(edge[1]))

        ids = [
            self.city_tuple_to_edge_id_dict[city_name][succ]
            for succ in succs
            if edge[0] != succ[1]
        ]

        return ids

    def plot_full_map(self, city_name: str, show: bool = True):
        """Plots the full map of a given city.

        Args:
            city_name (str): City name.
            show (bool, optional): Show plot. Defaults to True.
        """
        graph = self.graphs[city_name]

        self._plot_edges(graph.edges, city_name, show=show)

    def plot_lanes_by_id(
        self,
        lane_ids: List[int],
        city_name: str,
        color: str = "k",
        show: bool = True,
    ):
        """Plots all given lanes.

        Args:
            lane_ids (List[int]): Ids of lanes to plot.
            city_name (str): City name.
            color (str, optional): Color of the lanes. Defaults to "k".
            show (bool, optional): Show plot. Defaults to True.
        """
        graph = self.graphs[city_name]
        edges = list(graph.edges())
        edges = [edges[i] for i in lane_ids]

        self._plot_edges(edges, city_name, head_width=1, color=color, show=show)

    def _plot_edges(
        self,
        edges: List[Tuple[int, int]],
        city_name: str,
        head_width: float = 5,
        color: str = "k",
        show: bool = True,
    ):
        """Plot edges of the networkx graph as arrows.

        Args:
            edges (List[Tuple[int, int]]): Edges to plot.
            city_name (str): City name.
            head_width (float, optional): Arrow head width. Defaults to 5.
            color (str, optional): Color of the arrows. Defaults to "k".
            show (bool, optional): Show plot. Defaults to True.
        """
        graph = self.graphs[city_name]
        for edge in edges:
            s = edge[0]
            t = edge[1]

            x_s = graph.nodes[s]["x"]
            y_s = graph.nodes[s]["y"]
            x_t = graph.nodes[t]["x"]
            y_t = graph.nodes[t]["y"]

            plt.arrow(
                x_s, y_s, x_t - x_s, y_t - y_s, head_width=head_width, color=color
            )

        if show:
            plt.gca().set_aspect("equal")
            plt.show()

