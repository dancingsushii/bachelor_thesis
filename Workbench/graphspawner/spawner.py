import networkx as nx
import os
from utils import add_channel,pickle_graph,write_spawn_conf

### Defines the graph structure you will spawn ###
### Pickles it for later processing on the nodes ###
### Changes the spawn file to contain the nodes ###

### Provide weight in satoshis (default value used before: 5000000 [5 000 000]) ###
### We create only one channel per two nodes! Directions = node balance in channel ###
### Channel capacity = sum of balances in both directions ###

GRAPHSPAWNER_FILE = os.path.join(os.path.dirname(__file__), "graphspawner.conf")

def main():
    graph = nx.DiGraph()

    if os.path.exists(GRAPHSPAWNER_FILE):
        with open(GRAPHSPAWNER_FILE, "r") as f:
            for line in f:
                if line.startswith("#"): continue
                if line.strip() == "": continue
                src, dst, src_dst_balance, dst_src_balance = line.strip().split(" ")
                add_channel(graph, src, dst, int(src_dst_balance), int(dst_src_balance))
    else:
        ## Default setup
        alice = "alice"
        bob = "bob"
        carol = "carol"

        add_channel(graph, src = alice, dst = bob, src_dst_balance = 25000, dst_src_balance = 75000)
        add_channel(graph, alice, carol, 60000, 25000)
        add_channel(graph, bob, carol, 10000, 30000)

    # the following lines store your graph and nodes, don't remove them #
    pickle_graph(graph)
    write_spawn_conf(list(graph.nodes))


if __name__ == "__main__":
    main()