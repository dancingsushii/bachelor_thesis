import os
import json

def get_node_id_to_alias_dict(sharepath: str = "/share") -> dict:
    """Copy this function into your plugin to create a dictionary from files stored in nodeid_translator.py"""
    node_id_to_alias_file = os.path.join(sharepath, "node_id_to_alias.json")
    with open(node_id_to_alias_file, "r") as f: 
        return dict(json.load(f))

def get_alias_to_node_id_dict(sharepath: str = "/share") -> dict:
    """Copy this function into your plugin to create a dictionary from files stored in nodeid_translator.py"""
    alias_to_node_id_file = os.path.join(sharepath, "alias_to_node_id.json")
    with open(alias_to_node_id_file, "r") as f: 
        return dict(json.load(f))