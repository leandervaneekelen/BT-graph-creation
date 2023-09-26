import glob
import os
import numpy as np
import fire
import sys
import pandas as pd
from lxml import etree as ET
import re
import json

from collections import defaultdict

from tqdm import tqdm
from pathlib import Path

from scipy.spatial import distance, Delaunay
from sklearn.neighbors import NearestNeighbors

from utils import parse_xml, make_closed_loop


class EdgeConfig:
    """
    This class decodes the edge definition arguments
    edge_def_tb_to_l: "radius-X" or "to-X-nn" where X is a integer --> how should tumor buds and lymphocytes be connected
    edge_def_tb_to_tb: "radius-X" or "to-X-nn" where X is a integer --> how should tumor buds be connected
    fully_connected: specify either 'all', 'tumorbuds' or 'lymphocytes' ('all' supersedes the other --edge-def* arguments)
    hierarchical:
    """

    def __init__(
        self,
        edge_def_tb_to_l=None,
        edge_def_l_to_tb=None,
        edge_def_tb_to_tb=None,
        fully_connected=None,
        other=None,
    ):
        self.edge_def_tb_to_l = self.decode(edge_def_tb_to_l)
        self.edge_def_tb_to_tb = self.decode(edge_def_tb_to_tb)
        self.edge_def_l_to_tb = self.decode(edge_def_l_to_tb)
        self.fully_connected = fully_connected
        self.other = other

    @property
    def fully_connected(self):
        return self._fully_connected

    @fully_connected.setter
    def fully_connected(self, fully_connected):
        if fully_connected is not None:
            assert fully_connected in ["all", "tumorbuds", "lymphocytes"]
        self._fully_connected = fully_connected

    @property
    def other(self):
        return self._other

    @other.setter
    def other(self, other):
        if other is not None:
            assert other.split("-")[0] in ["hierarchical", "delaunay"]
        self._other = other

    @property
    def edge_definitions(self) -> dict:
        """
        sets up a dictionary with the edge definitions
        {edge_type: [params]}
        """
        edge_def = {}
        if self.fully_connected:
            edge_def["fully_connected"] = self.fully_connected
        if self.other:
            edge_def["other_edge_fct"] = self.other
        # fully connected 'all' supersedes the other edge definitions
        if self.fully_connected != "all":
            if self.edge_def_tb_to_tb and self.fully_connected != "tumorbuds":
                edge_def["tb_to_tb"] = self.edge_def_tb_to_tb
            if self.edge_def_tb_to_l and self.fully_connected != "lymphocytes":
                edge_def["tb_to_l"] = self.edge_def_tb_to_l
            if self.edge_def_l_to_tb and self.fully_connected != "lymphocytes":
                edge_def["l_to_tb"] = self.edge_def_l_to_tb
        return edge_def

    @staticmethod
    def decode(edge_def):
        """
        Decodes the edge definition string from the command line input
        """
        if edge_def:
            if "radius" in edge_def:
                return ["radius", int(edge_def.split("-")[-1])]
            elif "-nn-cutoff" in edge_def:
                return [
                    "kNN",
                    int(edge_def.split("-")[1]),
                    int(edge_def.split("-")[-1]),
                ]
            elif "-nn" in edge_def:
                return ["kNN", int(edge_def.split("-")[1])]
            elif "closest-cutoff" in edge_def:
                return ["to_closest", int(edge_def.split("-")[-1])]
            elif "closest" in edge_def:
                return ["to_closest"]
            elif "delaunay" in edge_def:
                return ["delaunay"]
            else:
                print(
                    f'Invalid input. Choose from "radius-X", "to-X-nn", "to-X-nn-cutoff-X" and "fully-connected" (specify number instead of X)'
                )
                sys.exit()
        else:
            return None

    def __str__(self):
        d = {k: "".join([str(i) for i in v]) for k, v in self.edge_definitions.items()}
        return "-".join([f"{k}_{v}" for k, v in d.items()])


DEFAULT_SPACING = 0.242797397769517


class Graph:
    """
    Creates a graph object from a list of text files that all need to have the same ID

    file_id: slide_name
    file_path: path to the files (without the ending, e.g. folder/slide_name_output)
    spacing: spacing from ASAP (list, e.g [0.24, 0.24])
    roi: optional (x1,y1,x2,x2) location describing the location of a region of interest.
         Only nodes within the ROI will be used to construct the graph.
    edge_config: EdgeConfig object
    """

    def __init__(
        self,
        file_id: str,
        file_path: str,
        spacing: tuple = None,
        roi: tuple = None,
        edge_config: EdgeConfig = None,
        csv_path: str = None,
    ) -> None:
        # print(f'Creating graph for id {file_id}.')
        self.file_path = file_path
        self.file_id = file_id
        self.spacing = (DEFAULT_SPACING, DEFAULT_SPACING) if spacing else spacing
        self.roi = roi
        self.node_feature_csv = csv_path
        self.edge_feature_names = []  # will get added during self.add_edges()

        self.xml_data = parse_xml(self.file_path)
        self.node_dict = self.get_node_dict()

        # set up the edges
        self.edge_config = edge_config.edge_definitions
        self.edge_dict = {}
        self.add_edges()

    # *********** properties ***********
    @property
    def node_feature_csv(self) -> dict:
        return self._node_feature_csv

    @node_feature_csv.setter
    def node_feature_csv(self, csv_path):
        if csv_path:
            df = pd.read_csv(csv_path, index_col=0).drop(
                "filename", axis=1, errors="ignore"
            )
            self._node_feature_csv = pd.DataFrame.to_dict(df, orient="index")
        else:
            self._node_feature_csv = None

    @property
    def xy_all_nodes(self) -> dict:
        return {
            node_id: (node_attrib["x"], node_attrib["y"])
            for node_id, node_attrib in self.node_dict.items()
        }

    @property
    def node_dict_per_label(self, label) -> dict:
        """
        Return the node_dict sorted by label.
        """
        node_dict_per_label = defaultdict(dict)
        for node_id, node_attrib in self.node_dict.items():
            label = node_attrib["type"]
            node_dict_per_label[label][node_id] = node_attrib
        return node_dict_per_label

    # *********** setting up the nodes ***********

    def _center_coordinates(self, coordinates):
        """
        Change the origin of the coordinates from the top left of the WSI
        to the top left of the ROI.
        """

        assert len(coordinates) == 2
        x1, y1, _, _ = self.roi
        coord = np.array(
            [
                coordinates[0] - x1,
                coordinates[1] - y1,
            ]
        )
        assert min(coord) >= 0
        if self.spacing:
            coord *= self.spacing
        return coord

    def _is_within_roi(self, coordinates) -> bool:
        assert len(coordinates) == 2
        x1, y1, x2, y2 = self.roi
        x, y = coordinates
        return (x1 <= x <= x2) and (y1 <= y <= y2)

    def get_node_dict(self):
        node_dict = defaultdict(dict)
        offset = 0

        # Populate nodes
        for label, coords in self.xml_data.items():
            # Filter points outside of roi
            if self.roi:
                coords = [
                    self._center_coordinates(c)
                    for c in coords
                    if self._is_within_roi(c)
                ]

            # Add x,y,c
            for i, coords_ in enumerate(coords):
                node_dict[i + offset] = {
                    "type": label,
                    "x": coords_[0],
                    "y": coords_[1],
                }

            # Add optional features from csv file (if present)
            # TODO: how do I know the node_id a priori if this is only defined in situ above?
            if self.node_feature_csv is not None:
                print(self.file_id)
                if len(self.node_feature_csv) == 0:
                    print(
                        f"No node features present in csv file for {self.file_id}. Skipping file."
                    )
                    return {}
                else:
                    node_dict = {
                        node_id: {**features, **self.node_feature_csv[node_id]}
                        for node_id, features in node_dict.items()
                    }
            offset += len(coords)

        return node_dict

    # *********** adding edges ***********
    def add_edges(self):
        """
        Adds all the edge to self.edge_dict in this format {edge_id: {'feature name 1': feature_value1, ...}, ...}
        The edge ID is the sorted string concatenation of the two node ids.
        The edge id of node 5 and node 0 is therefore '05'
        """
        # get all the edges
        if self.edge_config is not None and len(self.edge_config) > 0:
            for edge_type, param_list in self.edge_config.items():
                # edge_fct can be {'fully_connected', 'tb_to_tb', 'tb_to_l', 'l_to_tb' 'hierachical'}
                eval(f"self.{edge_type}")(param_list)

    def update_edge_dict(self, coo_matrix, edge_features, feature_name):
        """
        Updates self.edge_dict based on the coo_matrix and the edge_features list for a specific features (feature_name)

        The edge ID is the sorted string concatenation of the two node ids.
        The edge id of node 5 and node 0 is therefore '0_5'
        """
        # update the dictionary
        for edge, feature in zip(coo_matrix, edge_features):
            edge_id = [str(i) for i in sorted(edge)]
            edge_id_str = "_".join(edge_id)
            # if the edge already exists, just add the edge features
            if edge_id_str in self.edge_dict.keys():
                assert feature_name not in self.edge_dict[edge_id_str]
                self.edge_dict[edge_id_str][feature_name] = feature
            # if the edge does not exist, add it plus the feature
            else:
                self.edge_dict[edge_id_str] = {feature_name: feature}

    # *********** edge insertion functions ***********
    def tb_to_l(self, param_list):
        edge_fct = param_list[0]
        # add the edges
        if len(param_list) > 1:
            param_list = param_list[1:]
            eval(f"self.{edge_fct}")(self.xy_tb_nodes, self.xy_lymph_nodes, param_list)
        else:
            eval(f"self.{edge_fct}")(self.xy_tb_nodes, self.xy_lymph_nodes)

    def tb_to_tb(self, param_list):
        edge_fct = param_list[0]
        # add the edges
        if len(param_list) > 1:
            param_list = param_list[1:]
            eval(f"self.{edge_fct}")(self.xy_tb_nodes, self.xy_tb_nodes, param_list)
        elif edge_fct == "delaunay":
            self.delaunay(self.xy_tb_nodes)
        else:
            eval(f"self.{edge_fct}")(self.xy_tb_nodes, self.xy_tb_nodes)

    def l_to_tb(self, param_list):
        edge_fct = param_list[0]
        # add the edges
        if len(param_list) > 1:
            param_list = param_list[1:]
            eval(f"self.{edge_fct}")(self.xy_lymph_nodes, self.xy_tb_nodes, param_list)
        else:
            # add the edges
            eval(f"self.{edge_fct}")(self.xy_lymph_nodes, self.xy_tb_nodes)

    def fully_connected(self, params):
        # params should either be 'all', 'tumorbuds' or 'lymphocytes'
        assert params in ["all", "tumorbuds", "lymphocytes"]
        node_dict = {
            "all": self.xy_all_nodes,
            "tumorbuds": self.xy_tb_nodes,
            "lymphocytes": self.xy_lymph_nodes,
        }
        self.fully_connect(node_dict[params])

    def other_edge_fct(self, params):
        if params == "hierarchical":
            self.hierarchical()
        elif "hierarchical-cutoff" in params:
            cutoff = int(params.split("-")[-1])
            self.hierarchical(cutoff=cutoff)
        elif params == "delaunay":
            self.delaunay(self.xy_all_nodes)

    def hierarchical(self, cutoff=None):
        # connect lymphocytes to closest TB
        self.to_closest(self.xy_lymph_nodes, self.xy_tb_nodes, cutoff=cutoff)
        # fully connect lymphocytes that are connected to the same bud
        tb_ids_sub_graphs = {n: [] for n in self.xy_tb_nodes.keys()}
        sub_graphs = [(int(i) for i in e.split("_")) for e in self.edge_dict.keys()]
        for e_from, e_to in sub_graphs:
            if e_from in tb_ids_sub_graphs.keys():
                tb_ids_sub_graphs[e_from].append(e_to)
            elif e_to in tb_ids_sub_graphs.keys():
                tb_ids_sub_graphs[e_to].append(e_from)
        for tb_id, lymph_list in tb_ids_sub_graphs.items():
            self.fully_connect({i: self.xy_lymph_nodes[i] for i in lymph_list})
        # fully connect tumorbuds
        self.fully_connected("tumorbuds")

    def delaunay(self, node_dict):
        features_dict = {}
        points = np.array([list(i) for i in node_dict.values()])

        if len(points) < 4:
            return

        tri = Delaunay(points)
        for tr in tri.vertices:
            for i in range(3):
                edge_idx0 = tr[i]
                edge_idx1 = tr[
                    (i + 1) % 3
                ]  # to always get the next pointidx in the triangle formed by delaunay
                if min(node_dict.keys()) == 0:
                    edge = tuple(sorted([edge_idx0, edge_idx1]))
                else:
                    edge = tuple(
                        sorted(
                            [
                                edge_idx0 + min(node_dict.keys()),
                                edge_idx1 + min(node_dict.keys()),
                            ]
                        )
                    )
                # check the couple of points hasn't already been visited from the other side (= starting from the other point)
                # if yes then continue because already in the array
                if edge in features_dict.keys():
                    continue

                features_dict[edge] = distance.euclidean(
                    tri.points[edge_idx0], tri.points[edge_idx1]
                )

        # update the dictionary
        self.update_edge_dict(
            coo_matrix=features_dict.keys(),
            edge_features=features_dict.values(),
            feature_name="distance",
        )

    def fully_connect(self, node_dict):
        dict_ids = list(node_dict.keys())
        # features_dict = {}
        # for i in range(len(node_dict)):
        #     for j in range(i + 1, len(node_dict)):
        #         n1 = node_dict[dict_ids[i]]
        #         n2 = node_dict[dict_ids[j]]
        #         d = distance.euclidean(n1, n2)
        #         edge = tuple(sorted([dict_ids[i], dict_ids[j]]))
        #         features_dict[edge] = d
        features_dict = {
            tuple(sorted([dict_ids[i], dict_ids[j]])): distance.euclidean(
                node_dict[dict_ids[i]], node_dict[dict_ids[j]]
            )
            for i in range(len(node_dict))
            for j in range(i + 1, len(node_dict))
        }
        # update the dictionary
        self.update_edge_dict(
            coo_matrix=features_dict.keys(),
            edge_features=features_dict.values(),
            feature_name="distance",
        )

    def to_closest(self, from_dict, to_dict, params=None, cutoff=None):
        if params:
            cutoff = params.pop()
        features_dict = {}
        # calculate the distances
        if len(from_dict) > 0 and len(to_dict) > 0:
            for id_f, xy_f in from_dict.items():
                dist_list = [
                    distance.euclidean(xy_f, xy_t) for xy_t in to_dict.values()
                ]
                d = min(dist_list)
                id_t = list(to_dict.keys())[dist_list.index(d)]
                edge = tuple(sorted([id_f, id_t]))
                if edge not in features_dict.keys():
                    if cutoff:
                        if d < cutoff:
                            features_dict[edge] = d
                    else:
                        features_dict[edge] = d

        # update the dictionary
        self.update_edge_dict(
            coo_matrix=features_dict.keys(),
            edge_features=features_dict.values(),
            feature_name="distance",
        )

    def radius(self, center_dict, perimeter_dict, x):
        assert len(x) == 1
        x = x.pop()
        features_dict = {}
        # calculate the distances
        if len(center_dict) > 0 and len(perimeter_dict) > 0:
            for id_c, xy_c in center_dict.items():
                for id_p, xy_p in perimeter_dict.items():
                    # we don't want self loops
                    if id_c == id_p:
                        continue
                    d = distance.euclidean(xy_c, xy_p)
                    edge = tuple(sorted([id_c, id_p]))
                    # if d < x add edge
                    if d <= x and edge not in features_dict.keys():
                        features_dict[edge] = d

        # update the dictionary
        self.update_edge_dict(
            coo_matrix=features_dict.keys(),
            edge_features=features_dict.values(),
            feature_name="distance",
        )

    def kNN(self, center_dict, perimeter_dict, param_list, distance_metric="euclidean"):
        # set up k
        k = orig_k = param_list[0]
        cutoff = None
        if len(param_list) > 1:
            cutoff = param_list[1]

        # set-up in format for NearestNeighbors
        perimeter_keys = sorted(perimeter_dict.keys())
        center_keys = sorted(center_dict.keys())
        training_set = [perimeter_dict[i] for i in perimeter_keys]
        test_set = [center_dict[i] for i in center_keys]

        # if we are compare the same two sets, the first match will always be the point itself --> k += 1
        if training_set == test_set:
            k += 1
        # if #samples > k, set k to number of samples
        if k > len(training_set):
            k = len(training_set)

        features_dict = {}
        # only insert edges if we have elements in the lists
        if len(training_set) > 0 and len(test_set) > 0:
            neigh = NearestNeighbors(n_neighbors=k, metric=distance_metric)
            neigh.fit(training_set)

            distances_list, match_list = neigh.kneighbors(
                test_set, k, return_distance=True
            )
            for ind1, (indices, distances) in enumerate(
                zip(match_list, distances_list)
            ):
                for ind2, d in zip(indices, distances):
                    # ignore self matches and check for duplicates
                    # get the actual node ids
                    node_ind1 = center_keys[ind1]
                    node_ind2 = perimeter_keys[ind2]
                    edge = tuple(sorted([node_ind1, node_ind2]))
                    if node_ind1 != node_ind2 and edge not in features_dict.keys():
                        if cutoff is None:
                            features_dict[edge] = d
                        elif d <= cutoff:
                            features_dict[edge] = d

        # update the dictionary
        self.update_edge_dict(
            coo_matrix=features_dict.keys(),
            edge_features=features_dict.values(),
            feature_name="distance",
        )

    # *********** gxl creation ***********
    def sanity_check(self):
        node_ids = self.node_dict.keys()
        edge_ids = self.edge_dict.keys()

        # make sure all edges point to an existing node
        nodes_in_edges = [i.split("_") for i in edge_ids]
        nodes_in_edges = set(
            [int(val) for sublist in nodes_in_edges for val in sublist]
        )
        assert len(nodes_in_edges - set(node_ids)) == 0

    def get_gxl(self):
        """
        returns the xml-tree for the gxl file
        """
        print(f"Creating gxl tree for {self.file_id}.")
        self.sanity_check()
        type_dict = {"str": "string", "int": "int", "float": "float"}

        # initiate the tree
        xml_tree = ET.Element("gxl")

        # add the graph level info
        graph_attrib = {
            "id": self.file_id,
            "edgeids": "false",
            "edgemode": "undirected",
        }
        graph_gxl = ET.SubElement(xml_tree, "graph", graph_attrib)

        # add roi coordinates to gxl
        roi_gxl = ET.SubElement(xml_tree, "roi-coordinates")
        for i, coord in enumerate(make_closed_loop(*self.roi)):
            coor_attrib = {"Order": str(i), "X": str(coord[0]), "Y": str(coord[1])}
            _ = ET.SubElement(roi_gxl, "Coordinate", attrib=coor_attrib)

        # add the nodes
        for node_id, node_attrib in self.node_dict.items():
            node_gxl = ET.SubElement(graph_gxl, "node", {"id": "_{}".format(node_id)})
            for attrib_name, attrib_value in node_attrib.items():
                attrib_gxl = ET.SubElement(node_gxl, "attr", {"name": attrib_name})
                t = re.search(r"(\D*)", type(attrib_value).__name__).group(1)
                attrib_val_gxl = ET.SubElement(attrib_gxl, type_dict[t])
                attrib_val_gxl.text = str(attrib_value)

        # add the edges
        for edge_id, (edge_name, edge_attrib) in enumerate(self.edge_dict.items()):
            from_, to_ = edge_name.split("_")
            edge_gxl = ET.SubElement(
                graph_gxl, "edge", {"from": f"_{from_}", "to": f"_{to_}"}
            )
            for attrib_name, attrib_value in edge_attrib.items():
                attrib_gxl = ET.SubElement(edge_gxl, "attr", {"name": attrib_name})
                t = re.search(r"(\D*)", type(attrib_value).__name__).group(1)
                attrib_val_gxl = ET.SubElement(attrib_gxl, type_dict[t])
                attrib_val_gxl.text = str(attrib_value)

        # e = ET.dump(xml_tree)
        return xml_tree


class GxlFilesCreator:
    """
    Creates the xml trees from the text files with the coordinates
    """

    def __init__(
        self,
        files_to_process: list,
        edge_config: EdgeConfig,
        spacings: dict = None,
        rois: pd.DataFrame = None,
        normalize: bool = False,
        node_feature_csvs: str = None,
        datasplit_dict: dict = None,
    ):
        """
        files_to_process: list of paths to the files that should be processed
        spacings: dictionary that contains the spacing for each WSI (read from the spacing.json)
        rois: optional dataframe containing (x1,y1,x2,y2) coordinates describing the ROI within to build graphs
        edge_config: EdgeConfig object
        node_feature_csvs: path to folder with csv files with additional node features (e.g. ImageNet features)
        """
        self.files_to_process = files_to_process
        self.edge_config = edge_config
        self.spacings = spacings
        self.rois = rois
        self.normalize = normalize
        self.matched_csv = node_feature_csvs
        self.invalid_files = []
        self.datasplit_dict = datasplit_dict

    @property
    def matched_csv(self) -> dict:
        return self._matched_csv

    @matched_csv.setter
    def matched_csv(self, node_feature_csvs):
        self._matched_csv = None
        if node_feature_csvs:
            file_ids = [
                os.path.basename(f).split("_asap")[0] for f in self.files_to_process
            ]
            self._matched_csv = {
                file_id: csv
                for file_id in file_ids
                for csv in node_feature_csvs
                if file_id in csv
            }

    def check_files(self):
        # remove files that don't have a matching spacing and csv entry from self.files_to_process, and vice versa
        # TODO: return list of missing files
        # TODO: clean up this filtering because it's a mess
        file_ids_process = [Path(f).stem for f in self.files_to_process]
        if self.matched_csv is not None:
            common_ids = self._match_and_diff(
                set(self.matched_csv.keys()), set(file_ids_process)
            )
            self.files_to_process = [
                f for f in self.files_to_process if Path(f).stem in common_ids
            ]
        if self.spacings is not None:
            common_ids = self._match_and_diff(
                set(self.spacings.keys()), set(file_ids_process)
            )
            self.files_to_process = [
                f for f in self.files_to_process if Path(f).stem in common_ids
            ]
        if self.rois is not None:
            common_ids = self._match_and_diff(
                set(self.rois.keys()), set(file_ids_process)
            )
            self.files_to_process = [
                f for f in self.files_to_process if Path(f).stem in common_ids
            ]
        if self.matched_csv is not None:
            self._matched_csv = {
                file_id: csv
                for file_id, csv in self.matched_csv.items()
                if file_id in common_ids
            }
        if self.spacings is not None:
            self.spacings = {
                file_id: spacing
                for file_id, spacing in self.spacings.items()
                if file_id in common_ids
            }
        if self.rois is not None:
            self.rois = {
                file_id: roi
                for file_id, roi in self.rois.items()
                if file_id in common_ids
            }

    def _match_and_diff(self, set1, set2) -> set:
        common_ids_csv_process = set1 & set2
        self.invalid_files += sorted(list(set(set1).symmetric_difference(set(set2))))
        return common_ids_csv_process

    def get_xml(self, file_id, file_path):
        return self.get_graph(file_id, file_path).get_gxl()

    def get_graph(self, file_id, file_path):
        spacing = self.spacings[file_id] if self.spacings else None
        roi = self.rois[file_id] if self.rois else None
        csv_path = self.matched_csv[file_id] if self.matched_csv else None
        return Graph(
            file_id=file_id,
            file_path=file_path,
            spacing=spacing,
            roi=roi,
            edge_config=self.edge_config,
            csv_path=csv_path,
        )

    def save_gxls(self, output_path: str):
        # create output folder if it does not exist
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        if self.datasplit_dict is not None:
            folders_to_create = [
                os.path.join(output_path, split, cls)
                for split, d in self.datasplit_dict.items()
                for cls in d.keys()
            ]
            _ = [os.makedirs(p) for p in folders_to_create if not os.path.isdir(p)]

        # save the xml trees
        print(f"Saving gxl files to {output_path}")
        if self.datasplit_dict is not None:
            file_id_to_folder = {
                file_id: [split, cls]
                for split, d in self.datasplit_dict.items()
                for cls, file_ids in d.items()
                for file_id in file_ids
            }

        self.check_files()  # make sure we only have files that have a corresponding csv / spacing (if provided)
        files_dict = {Path(f).stem: f for f in self.files_to_process}
        # remove files that are not in the datasplit dict
        if self.datasplit_dict is not None:
            files_dict = {
                f_id: path
                for f_id, path in files_dict.items()
                if f_id in file_id_to_folder.keys()
            }

        for file_id, file_path in tqdm(files_dict.items()):
            xml_tree = self.get_xml(file_id, file_path)

            if self.datasplit_dict is None:
                outfolder = output_path
            else:
                try:
                    outfolder = os.path.join(
                        output_path, "/".join(file_id_to_folder[file_id])
                    )
                except KeyError:
                    self.invalid_files.append(file_id)
                    continue
            # save the file
            with open(os.path.join(outfolder, file_id + ".gxl"), "wb") as f:
                f.write(ET.tostring(xml_tree, pretty_print=True))

        self._save_log(output_path)

    def _save_log(self, output_folder):
        # Save the files where we had a missing spacing, xml or csv file (if present)
        if len(self.invalid_files) > 0:
            with open(
                os.path.join(
                    output_folder,
                    f"{os.path.basename(output_folder)}_invalid_file_ids.txt",
                ),
                "w",
            ) as f:
                f.write("\n".join(sorted(self.invalid_files)))


def make_gxl_dataset(
    asap_xml_files_folder: str,
    output_folder: str,
    edge_def_tb_to_l: str = None,
    edge_def_tb_to_tb: str = None,
    edge_def_l_to_tb: str = None,
    fully_connected: str = None,
    spacing_json: str = None,
    roi_json: str = None,
    node_feature_csvs: str = None,
    split_json: str = None,
    other_edge_fct: str = None,
    overwrite: bool = False,
):
    """
    INPUT
     - `--asap_xml_files_folder`: path to the folder with the coordinates xml files
     - `--edge-def-tb-to-l`, `--l-to-tb` and `--edge-def-tb-to-tb` have the following options:
       - `radius-x`: connect elements in radius X (in micrometer)
       - `to-X-nn`: connect to k closest elements where X is the number of neighbours
       - `to-X-nn-cutoff-Y`: connect to k closest elements where X is the number of neighbours, if the
         distance between them is smaller than Y (micrometers)
       - `to-closest`: adds edge to the closest neighbour
       - `to-closest-cutoff-X`: adds edge to the closest neighbour, if distance is smaller than X (micrometers)
       - `delaunay`: only works for tumor bud to tumor bud connection. Connects them based on Delaunay triangulation
     - `--fully-connected`: supersedes the `--edge-def*`. Following options:
       - `all`: fully connected graph
       - `lymphocytes`: only the lymphocytes are fully connected
       - `tumorbuds`: only the tumorbuds are fully connected
     - `--other-edge-fct`: supersedes the `--edge-def*`. Following options:
       - `hierarchical`: creates a graph where the tumor buds are fully connected, and the T-cells are
         connected to the closest tumor bud.
       - `delaunay`: performs delaunay triangulation (regardless of node label)
     - `--output-folder`: path to where output folder should be created
     - `--node-feature-csvs`: optional. Path to folder with csv files that contain additional node features.
       The first column needs to have the node index number. The headers will be used as the feature name.
       If there is a column named "filename", it will be dropped.
     - `--spacing-json`: optional. Path to json file that contains the spacing for each whole slide image.
        It is needed to compute the distance between elements. (default is 0.242797397769517).
     - `--roi-json`: optional. Path to JSON that contains (x1,y1,x2,y2) tuples per whole slide image.
        Points outside of the ROI are not used for constructing the graph.
     - `overwrite`: optional. Set if you want existing gxl files to be overwritten. Default is False

    OUTPUT
    One gxl file per hotspot, which contains the graph (same structure as the gxl files from the IAM Graph Databse)
    """
    # get the edge definitions
    edge_def_config = EdgeConfig(
        edge_def_tb_to_l=edge_def_tb_to_l,
        edge_def_tb_to_tb=edge_def_tb_to_tb,
        edge_def_l_to_tb=edge_def_l_to_tb,
        fully_connected=fully_connected,
        other=other_edge_fct,
    )

    subfolder = str(edge_def_config) if edge_def_config else "no_edges"
    output_path = os.path.join(output_folder, subfolder)

    # read the spacing json
    if spacing_json:
        spacing_json = r"{}".format(spacing_json)
        with open(spacing_json) as data_file:
            spacings = json.load(data_file)
    else:
        spacings = (
            None  # spacing of 0.242797397769517 will be used (default in Graph())
        )

    # Read the ROI csv
    if roi_json:
        with open(roi_json) as file:
            rois = json.load(file)
    else:
        rois = None

    # get a list of all the xml files to process
    if not os.path.isdir(asap_xml_files_folder):
        print(f"Folder {asap_xml_files_folder} does not exist. Exiting...")
        sys.exit(-1)
    files_to_process = glob.glob(os.path.join(asap_xml_files_folder, "*.xml"))
    # files_to_process = list(set([re.search(r'(.*)_coordinates', f).group(1) for f in all_files]))
    if len(files_to_process) == 0:
        print(f"No files found to process! Exiting...")
        sys.exit(-1)

    # read the split json (if present)
    if split_json is not None:
        with open(split_json) as data_file:
            datasplit_dict = json.load(data_file)
    else:
        datasplit_dict = None

    # get a list of the csv files
    if node_feature_csvs is not None:
        if os.path.isdir(node_feature_csvs):
            node_feature_csvs = glob.glob(os.path.join(node_feature_csvs, "*.csv"))
        else:
            print(
                f"Folder {node_feature_csvs} with node feature csv does not exist. Exiting."
            )
            sys.exit(-1)

    # get list of existing files
    if overwrite:
        print("Existing files will be overwritten!")
    else:
        existing_gxl = [path.stem for path in Path(output_folder).glob("*.gxl")]
        files_to_process_id = [Path(file).stem for file in files_to_process]
        files_to_process = [
            file_path
            for file_id, file_path in zip(files_to_process_id, files_to_process)
            if file_id not in existing_gxl
        ]

    # Create the gxl files
    GxlFilesCreator(
        files_to_process=files_to_process,
        spacings=spacings,
        rois=rois,
        edge_config=edge_def_config,
        node_feature_csvs=node_feature_csvs,
        datasplit_dict=datasplit_dict,
    ).save_gxls(output_path=output_path)


if __name__ == "__main__":
    fire.Fire(make_gxl_dataset)
