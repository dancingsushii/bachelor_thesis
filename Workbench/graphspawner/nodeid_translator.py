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

def main():
    """ Collect node_ids and node_aliases and write them into two dictionary files for further usage """
    
    # go through folders in share
    sharepath = os.path.dirname(__file__).replace("graphspawner", "share")
    share_contents = os.listdir(sharepath)
    
    node_id_to_alias = dict()
    alias_to_node_id = dict()
    
    for path in share_contents:
        
        fullpath = os.path.join(sharepath, path)

        if ".regtest.node" not in path: continue
        if not os.path.isdir(fullpath): continue
        
        node_id = None
        alias = path.strip().replace(".regtest.node", "")
        
        # read node configuration file
        node_config = os.path.join(fullpath, "lightning-peer.conf")
        if not os.path.exists(node_config): 
            print(f"ERROR: {node_config} not found when creating node_id <--> alias dictionaries!")
            exit(-1)
        
        with open(node_config, "r") as f:
            for line in f:
                if line.strip() == "": continue
                key, value = line.strip().split("=")
                if key == "NODE_ID": 
                    node_id = value.strip()
                    break
        
        # Write dictionaries
        node_id_to_alias[node_id] = alias
        alias_to_node_id[alias] = node_id
    
    # store dictionaries for every node to access
    node_id_to_alias_file = os.path.join(sharepath, "node_id_to_alias.json")
    with open(node_id_to_alias_file, "w") as f:
        json.dump(node_id_to_alias, f)
    
    alias_to_node_id_file = os.path.join(sharepath, "alias_to_node_id.json")
    with open(alias_to_node_id_file, "w") as f:
        json.dump(alias_to_node_id, f)


if __name__ == "__main__":
    main()