import networkx as nx
import os

def main():
    print("Starting to process the stored networkx graph...")
    # read given graph data from the share folder
    graphfile = "/share/graph_pickle.gpickle.gz"
    if not os.path.exists(graphfile):
        print(f"ERROR: graph not pickled under {graphfile}, file not found!")
        exit(-1)
    graph: nx.DiGraph = nx.read_gpickle(graphfile)
    # who am I? --> get pod name
    with open("/etc/hostname", "r") as f:
        my_node = f.readline().strip().replace(".regtest.node", "")
    # find who I have connections to and add them to $CONN_LIST env variable
    connections = dict(graph[my_node])
    graph_config_dir = "/root/run/graphlib/configs"
    if not os.path.exists(graph_config_dir): os.mkdir(graph_config_dir)
    for peer_alias in connections:
        # add connection to system variable CHAN_LIST
        src_dst_balance = connections[peer_alias]["src_dst_balance"]
        dst_src_balance = connections[peer_alias]["dst_src_balance"]
        our_funding = src_dst_balance + dst_src_balance
        # create dedicated files for each peers configs
        peer_graph_config_file = os.path.join(graph_config_dir, f"{peer_alias}.conf")
        with open(peer_graph_config_file, "w") as f:
            f.write(f"PEER_ALIAS={peer_alias}\n")
            f.write(f"OUR_FUNDING={our_funding}\n")
            f.write(f"THEIR_FUNDING={dst_src_balance}")


if __name__ == "__main__":
    main()