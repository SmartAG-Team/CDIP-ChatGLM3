import json

def load_data(source_file,reference_file):
    with open(source_file, "r", encoding="utf-8") as json_file:
        data_source = json.load(json_file)
    with open(reference_file, "r", encoding="utf-8") as json_file:
        data_reference = json.load(json_file)
    return data_source, data_reference
