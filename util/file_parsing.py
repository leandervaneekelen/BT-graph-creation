import os
import xml.etree.ElementTree as ET
from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation
from collections import defaultdict


def parse_xml(file_path) -> dict:
    wsa = WholeSlideAnnotation(annotation_path=file_path)
    annos_per_label = wsa.annotations_per_label
    coords_per_label = defaultdict(list)
    for label, annos in annos_per_label.items():
        for anno in annos:
            if anno.type == "polygon":
                coords = tuple(map(tuple, anno.coordinates))
            elif anno.type == "point":
                coords = tuple(anno.coordinates)
            else:
                raise NotImplementedError
        coords_per_label[label] = coords
    return coords_per_label

    # tree = ET.parse(file_path)
    # root = tree.getroot()

    # groups_colours = {i.attrib["Name"]: i.attrib["Color"] for i in root.iter("Group")}
    # groups = ["hotspot", "lymphocytes", "tumorbuds", "lymphocytesR", "tumorbudsR"]
    # annotations_elements = {g: [] for g in groups}

    # for i in root.iter("Annotation"):
    #     annotations_elements[i.attrib["PartOfGroup"]].append(i)

    # annotations = {g: [] for g in groups}
    # for group, element_list in annotations_elements.items():
    #     for element in element_list:
    #         if element.attrib["Type"] == "Dot":
    #             annotations[group].append(
    #                 [
    #                     [float(i.attrib["X"]), float(i.attrib["Y"])]
    #                     for i in element.iter("Coordinate")
    #                 ][0]
    #             )
    #         else:
    #             if group in ["lymphocytes", "tumorbuds"]:
    #                 group = "rectangles_" + group
    #             annotations[group].append(
    #                 [
    #                     [float(i.attrib["X"]), float(i.attrib["Y"])]
    #                     for i in element.iter("Coordinate")
    #                 ]
    #             )

    # return annotations
