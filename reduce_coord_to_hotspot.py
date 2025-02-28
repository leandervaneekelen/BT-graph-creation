import glob
import os
import numpy as np
from lxml import etree as ET
import argparse
import re
import shutil
import pandas as pd

from coord_to_xml import create_asap_xml
from xml_to_txt_file import process_xml_files


def setup_output_folders(output_path):
    # set / create the output
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    # make the output folders
    xml_output = os.path.join(output_path, "asap_xml")
    if not os.path.isdir(xml_output):
        os.mkdir(xml_output)
    txt_output = os.path.join(output_path, "coordinates_txt")
    if not os.path.isdir(txt_output):
        os.mkdir(txt_output)

    return xml_output, txt_output


def in_square(square_coordinates, point):
    assert len(square_coordinates) == 4 and len(point) == 2
    x = [i[0] for i in square_coordinates]
    y = [i[1] for i in square_coordinates]

    x_dim = min(x) <= point[0] <= max(x)
    y_dim = min(y) <= point[1] <= max(y)

    return x_dim and y_dim


def read_hotspot_xmls(hotspot_xmls):
    all_hotspots = {}
    for file_path in hotspot_xmls:
        if os.path.isfile(file_path):
            tree = ET.parse(file_path)
            root = tree.getroot()
            filename = os.path.basename(os.path.splitext(file_path)[0])
            filename = re.sub(" ", "_", filename)

            group = "hotspot"
            annotations_elements = [
                i for i in root.iter("Annotation") if i.attrib["PartOfGroup"] == group
            ]

            annotations = [
                [
                    [float(i.attrib["X"]), float(i.attrib["Y"])]
                    for i in element.iter("Coordinate")
                ]
                for element in annotations_elements
            ]

            all_hotspots[filename] = annotations
        else:
            print(f"File {file_path} does not exist.")
    return all_hotspots


def parse_hotspot_xml(hotspot_xmls, txt_output):
    # make txt files
    process_xml_files(hotspot_xmls, txt_output)
    # rename the hotspot txt files to add '_output' into them so all the files have the same name
    # hotspot_txts = glob.glob(os.path.join(txt_output, r'*hotspot*'))
    # for ht in hotspot_txts:
    #     if '_output' not in ht:
    #         regex = re.search('(.*)(_coordinates.*)', ht)
    #         shutil.move(ht, '{}_output{}'.format(regex.group(1), regex.group(2)))

    # read in the hotspot xml
    hotspots = read_hotspot_xmls(hotspot_xmls)
    return hotspots


def check_output(txt_output, xml_output, all_hotspots):
    # make sure all the hotspots were processed and that there are three txt files for each hotspot
    txt_files = [
        os.path.basename(f).split("_CD")[0] for f in glob.glob(txt_output + "/*.txt")
    ]
    txt_files_ids = set(txt_files)
    xml_files_ids = set(
        [os.path.basename(f).split("_CD")[0] for f in glob.glob(xml_output + "/*.xml")]
    )
    hotspot_ids = set([os.path.basename(f).split("_CD")[0] for f in all_hotspots])

    # check that there are three files for each hotspot
    text_files_unique = np.unique(txt_files, return_counts=True)
    not_three = ", ".join(
        [
            text_files_unique[0][i]
            for i, count in enumerate(text_files_unique[1])
            if count != 3
        ]
    )
    print(f"Not three output text files for files {not_three}")

    # check for missing files
    missing_txt_files = ", ".join(list(hotspot_ids - txt_files_ids))
    missing_xml_files = ", ".join(list(hotspot_ids - xml_files_ids))

    stats = f"# hotspots: {len(hotspot_ids)} | # text files: {len(txt_files_ids)} | # xml files: {len(xml_files_ids)}"
    missing_txt_msg = f"missing text files for hotspot(s) {missing_txt_files}"
    missing_xml_msg = f"missing xml files for hotspot(s) {missing_xml_files}"

    print(stats)
    print(missing_txt_msg)
    print(missing_xml_msg)

    # write the files with issues to a file
    with open(os.path.join(os.path.dirname(xml_output), "log.txt"), "w") as text_file:
        text_file.write("\n".join([stats, missing_txt_msg, missing_xml_msg]))


def create_hotspot_only_txt_files(
    txt_files_to_process,
    xml_output,
    txt_output,
    all_hotspots,
    overwrite=False,
    no_xml=False,
):
    output_text_files = []
    error_files = []
    for file in txt_files_to_process:
        print("Processing file {}".format(os.path.basename(file)))
        coord_in_hotspot = {}
        for group in ["lymphocytes", "tumorbuds"]:
            if os.path.basename(file) in all_hotspots.keys():
                hotspots = all_hotspots[os.path.basename(file)]

                # load the file
                file_path = f"{file}_coordinates_{group}.txt"

                if os.path.isfile(file_path):
                    coordinates = np.loadtxt(file_path)

                    # iterate over hotspot files
                    for h_ind, h in enumerate(hotspots):
                        if len(hotspots) == 1:
                            output_txt_file = os.path.join(
                                txt_output, os.path.basename(file_path)
                            )
                        else:
                            output_txt_file = os.path.join(
                                txt_output,
                                f"{os.path.basename(file)}_hotspot{h_ind}_coordinates_{group}.txt",
                            )
                        in_hotspot = [in_square(h, i) for i in coordinates]
                        coord_in_hotspot[group] = coordinates[in_hotspot]
                        to_save = coordinates[in_hotspot]
                        output_text_files.append(output_txt_file)
                        if not os.path.isfile(output_txt_file) or overwrite:
                            np.savetxt(output_txt_file, to_save, fmt="%.4f")
                        else:
                            print(
                                "The coordinates file {} already exists".format(
                                    output_txt_file
                                )
                            )
                else:
                    print("File {} does not exist. Continuing...".format(file_path))
                    error_files.append(file_path)
            else:
                print(f"No hotspot xml for file {os.path.basename(file)}")
                error_files.append(file)

    if not no_xml:
        create_asap_xml(txt_output, xml_output)

    # make sure all the hotspots were processed and that there are three txt files for each hotspot
    check_output(txt_output, xml_output, all_hotspots.keys())


MATCHED_EXCEL_INFO = {
    "wsi_col": "CD8 Filename",
    "xml_col": "Hotspot filename",
    "sheet_name": "BTS",
    "folder_col": "Folder",
}

if __name__ == "__main__":
    # Expects the hotspot files to have the same name as the matching coordinate files
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-hotspot-folder", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--coordinate-txt-files", type=str, required=True)
    parser.add_argument(
        "--overwrite", action="store_true", required=False, default=False
    )
    parser.add_argument("--no-xml", action="store_true", required=False, default=False)

    args = parser.parse_args()

    hotspot_path = args.xml_hotspot_folder

    output_path = args.output_folder

    xml_output, txt_output = setup_output_folders(output_path)

    coor_txt_files_path = os.path.join(
        args.coordinate_txt_files, r"*_coordinates_*.txt"
    )
    all_txt_files = glob.glob(coor_txt_files_path)
    txt_files_to_process = {
        os.path.basename(file): file
        for file in list(
            set([re.search(r"(.*)_coordinates", f).group(1) for f in all_txt_files])
        )
    }

    # get the hotspots (dict)
    set([os.path.basename(f) for f in txt_files_to_process])
    all_hotspot_ids = set(
        [
            str.split(os.path.basename(f), ".xml")[0]
            for f in glob.glob(os.path.join(hotspot_path, "*.xml"))
        ]
    )
    matched = all_hotspot_ids.intersection(set(txt_files_to_process.keys()))
    unmatched = all_hotspot_ids.symmetric_difference(set(txt_files_to_process.keys()))
    print(
        f"Following file-ids could not be matched (no corresponding hotspot or txt file): {unmatched}"
    )
    hotspot_files = [os.path.join(hotspot_path, f"{f}.xml") for f in matched]
    hotspots = parse_hotspot_xml(hotspot_files, txt_output)

    txt_files_to_process = [
        f for name, f in txt_files_to_process.items() if name in matched
    ]
    # create the hotspot asap and txt files
    create_hotspot_only_txt_files(
        txt_files_to_process,
        xml_output,
        txt_output,
        hotspots,
        args.overwrite,
        args.no_xml,
    )
