#!/usr/bin/env python3
# Plugin libraries
from pyln.client import Plugin, Millisatoshi, RpcError
from threading import Thread, Lock
from datetime import timedelta
from functools import reduce
import time
import uuid
from mpyc.runtime import mpc

# H&S libraries
# mathematical optimization software library for solving mixed-integer linear and quadratic optimization problems
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp




plugin = Plugin()
plugin.rebalance_stop = False


# Plugin initialization
@plugin.init()
def init(options, configuration, plugin):
    plugin.log("Hide & Seek rebalancing plugin initialized")

    # The listconfigs RPC command to list all configuration options, or with config only a selection.
    config = plugin.rpc.listconfigs()

    ## Collect all necessary information about current node
    # Check Lock Time Verify
    plugin.cltv_final = config.get("cltv-final")
    plugin.fee_base = Millisatoshi(config.get("fee-base"))
    plugin.fee_ppm = config.get("fee-per-satoshi")

    # TODO Maybe add other options needed
    plugin.threshold = int(options.get("rebalancing-threshold"))

    plugin.mutex = Lock()
    plugin.log(f"Plugin rebalance initialized with {plugin.fee_base} base / {plugin.fee_ppm} ppm fee  "
               f"cltv_final:{plugin.cltv_final}  "
               # f"maxhops:{plugin.maxhops}  "
               # f"msatfactor:{plugin.msatfactor}  "
               # f"erringnodes:{plugin.erringnodes}  "
               # f"getroute:{plugin.getroute.__name__}  "
               )

# TODO find out what is msatoshi, maxfeepercent, exemptfee
#%
@plugin.method("rebalance")
def rebalance(plugin, outgoing_scid, incoming_scid, msatoshi: Millisatoshi = None,
              retry_for: int = 60, maxfeepercent: float = 0.5,
              exemptfee: Millisatoshi = Millisatoshi(5000)):
    """Rebalancing of one particular channel of a node based on Hide & Seek protocol.

    White paper can be found here:
    https://arxiv.org/pdf/2110.08848.pdf
    """
    if msatoshi:
        msatoshi = Millisatoshi(msatoshi)

    # retry_for = int(retry_for)
    # maxfeepercent = float(maxfeepercent)
    # exemptfee = Millisatoshi(exemptfee)
    payload = {
        "outgoing_scid": outgoing_scid,
        "incoming_scid": incoming_scid,
        "msatoshi": msatoshi,
        "retry_for": retry_for,
        "maxfeepercent": maxfeepercent,
        "exemptfee": exemptfee
    }

    # here are the rpc instructions
    my_node_id = plugin.rpc.getinfo().get('id')
    outgoing_node_id = peer_from_scid(plugin, outgoing_scid, my_node_id, payload)
    incoming_node_id = peer_from_scid(plugin, incoming_scid, my_node_id, payload)
    get_channel(plugin, payload, outgoing_node_id, outgoing_scid, True)
    get_channel(plugin, payload, incoming_node_id, incoming_scid, True)
    out_ours, out_total = amounts_from_scid(plugin, outgoing_scid)
    in_ours, in_total = amounts_from_scid(plugin, incoming_scid)

    # MPC input
    secint = mpc.SecInt(16)

    # Each party enters an input. That is number of peers and channels to these peers.
    my_age = int(input('Enter your m and n: '))

    # List with one secint per party
    our_ages = mpc.input(secint(my_age))


    return s


# Linear program to solve for finding rebalancing
def LP_global_rebalancing(rebalancing_graph):

    # last_time = time.time()

    try:
        # Number of connected to this node peers
        # n = rebalancing_graph.number_of_nodes()
        n = len(plugin.rpc.listpeers())

        # Number of edges from this node. Careful with bidirectiona channels please!
        # m = rebalancing_graph.number_of_edges()
        m = len(plugin.rpc.listchannels())

        # TBC
        global list_of_nodes
        global list_of_edges
        # list_of_nodes = list(rebalancing_graph.nodes)
        list_of_nodes = list(plugin.rpc.listpeers())
        # list_of_edges = list(rebalancing_graph.edges)
        list_of_edges = list(plugin.rpc.listchannels())

        # Create a new model
        model = gp.Model("rebalancing-LP")

        # Create variables
        x = model.addMVar(shape=m, vtype=GRB.CONTINUOUS, name="x")

        # Set objective
        obj = np.zeros(m, dtype=float)

        # only channels where transactions failed contribute to the objective function
        # Do not know how to translate this?
        for edge_index in range(m):
            u, v = list_of_edges[edge_index]
            if 'objective function coefficient' in rebalancing_graph[u][v]:
                obj[edge_index] = rebalancing_graph[u][v]['objective function coefficient']

        model.setObjective(obj @ x, GRB.MAXIMIZE)

        ## adding constraints
        data = []
        row = []
        col = []
        rhs = np.zeros(2 * m + 2 * n)

        # constraint 1: respecting capacities: 0 <= f(u,v) <= m(u,v)
        # I.e. -f(u,v) <= 0 and f(u,v) <= m(u,v)
        # sequential code
        for edge_index in range(m):
            u,v = list_of_edges[edge_index]

        #     # -f(u,v) <= 0
            append_to_A(-1, edge_index, edge_index, data, row, col)
        #     # bound is zero, so no need to edit rhs
        #     # rhs[edge_index] = 0

        #     # f(u,v) <= m(u,v)
            append_to_A(1, m + edge_index, edge_index, data, row, col)
            rhs[m + edge_index] = rebalancing_graph[u][v]['flow_bound']
        print('done with constraint 1')

        # parallel code
        # 1a: -f(u,v) <= 0
        # inputs_1a = range(m)
        # with concurrent.futures.ProcessPoolExecutor(max_workers=num_of_conc_cores) as executor:
            # results_1a = executor.map(compute_costraint_1a, inputs_1a)  # results is a generator object

        # 1b: f(u,v) <= m(u,v)
        # inputs_1b = [(m, edge_index) for edge_index in range(m)]
        # with concurrent.futures.ProcessPoolExecutor(max_workers=num_of_conc_cores) as executor:
            # results_1b = executor.map(compute_costraint_1b, inputs_1b)  # results is a generator object

        # all_entries = [result for result in results_1a] + [result for result in results_1b]  # unpack generator

        # set bound vector
        # for edge_index in range(m):
            # u, v = list_of_edges[edge_index]
            # rhs[m + edge_index] = rebalancing_graph[u][v]['flow_bound']

        # print(f'done with constraint 1 in {duration_since(last_time)}')
        # last_time = time.time()

        # constraint 2: flow conservation: sum of in flows = some of out flows
        # ineq 2a: \sum_{out edges} f(u,v) - \sum_{in edges} f(v,u) <= 0
        # ineq 2b: \sum_{in edges} f(v,u) - \sum_{out edges} f(u,v) <= 0
        # bounds (rhs vector) are already set to zero from initialization
        # sequential code
        for i in range(n):
        #     # all bounds are zero, thus no need to edit rhs

             u = list_of_nodes[i]

             for edge in rebalancing_graph.out_edges(u):
                 edge_index = list_of_edges.index(edge)

        #         # ineq 2a: \sum_{out edges} f(u,v)
                 append_to_A(1, 2*m + i, edge_index, data, row, col)

        #         # ineq 2b: - \sum_{out edges} f(u,v)
                 append_to_A(-1, 2*m + n + i, edge_index, data, row, col)

             for edge in rebalancing_graph.in_edges(u):
                 edge_index = list_of_edges.index(edge)

        #         # ineq 2a: - \sum_{in edges} f(v,u)
                 append_to_A(-1, 2*m + i, edge_index, data, row, col)

        #         # ineq 2b: \sum_{in edges} f(v,u)
                 append_to_A(1, 2*m + n + i, edge_index, data, row, col)

        # parallel code
        # inputs_2 = [(n, m, node_index) for node_index in range(n)]
        # with concurrent.futures.ProcessPoolExecutor(max_workers=num_of_conc_cores) as executor:
            # results_1b = executor.map(compute_costraint_2, inputs_2)  # results is a generator object

        print('done with constraint 2')

        A_num_of_rows = 2 * m + 2 * n
        A_num_of_columns = m

        A = sp.csr_matrix((data, (row, col)), shape=(A_num_of_rows, A_num_of_columns))

        # Add constraints
        model.addConstr(A @ x <= rhs, name="matrix form constraints")

        # Optimize model
        model.optimize()

        try:
            print(x.X)
            print(f'Obj: {model.objVal}')

            flows = x.X
        except:
            # infeasible model, set all flows to zero
            print('model is infeasible, setting all flows to zero')
            flows = list(np.zeros(m, dtype=int))

        balance_updates = []

        # flow updates
        for edge_index in range(m):
            u, v = list_of_edges[edge_index]
            balance_updates.append((u, v, int(flows[edge_index])))
            # print(f'flow({u},{v}) = {int(flows[edge_index])}')

        return balance_updates

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')


# Helping method for LP solving
def append_to_A(d, r, c, data, row, col):
    # append single items or lists to data, row, col
    if type(d) != list:
        data.append(d)
        row.append(r)
        col.append(c)
    else:
        for i in range(len(d)):
            data.append(d[i])
            row.append(r[i])
            col.append(c[i])


# Method from plugin that returns peers from channel
def peer_from_scid(plugin, short_channel_id, my_node_id, payload):
    channels = plugin.rpc.listchannels(short_channel_id).get('channels')
    for ch in channels:
        if ch['source'] == my_node_id:
            return ch['destination']
    raise RpcError("rebalance", payload, {'message': 'Cannot find peer for channel: ' + short_channel_id})


# Method for getting all peers of a node.
def get_open_channels(plugin: Plugin):
    channels = []
    for peer in plugin.rpc.listpeers()["peers"]:
        for ch in peer["channels"]:
            if ch["state"] == "CHANNELD_NORMAL" and not ch["private"]:
                channels.append(ch)
    return channels


#plugin.add_option(
#    'greeting',
#    'This is a first implementation of a rebalancing plugin based on Hide & Seek protocol. '
#    'You can find all preliminaries and theoretical background at https://arxiv.org/pdf/2110.08848.pdf'
#)


# # Hide & Seek global variables
# global_rebalancing_threshold = 50
plugin.add_option(
    "rebalance-threshold",
    "50",
    "Global rebalancing threshold defined in the white paper"
    "Note: review that one later.",
    "string"
)

plugin.run()
