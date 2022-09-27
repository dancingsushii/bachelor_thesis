# mathematical optimization software library for solving mixed-integer linear and quadratic optimization problems
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import copy
import time
# mathematical optimization software library for solving mixed-integer linear and quadratic optimization problems
import gurobipy as gp
from gurobipy import GRB
# sparse array package for numeric data
import scipy.sparse as sp
# package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks
import networkx as nx
# import statistics
# import ast
# import pickle5 as pickle


# module provides a high-level interface for asynchronously executing callables.
import concurrent.futures

# For time evaluation of experiment
def duration_since(last_time):
    delta = time.time() - last_time  # duration in seconds
    if delta < 60:
        duration = delta
        measure = 'seconds'
    elif delta < 3600:
        duration = delta / 60
        measure = 'minutes'
    else:
        duration = delta / 3600
        measure = 'hours'

    duration = int(duration * 100) / 100  # truncate to two decimals
    return f'{duration} {measure}'

# Writing the results
def write_and_print(filename, text):
    f = open(filename, "a")
    f.write(text + '\n')
    f.close()
    print(text)

# Proceed with all transactions without rebalancing that is to execute them anyway.
def no_rebalancing(topology, transaction_list):
    # do all transactions that are possible

    successful_transactions = 0
    successful_volume = 0

    for transaction in transaction_list:
        successful_transaction, failed_at_channel, topology = execute_transaction(topology, transaction)

        if successful_transaction:
            successful_transactions += 1
            successful_volume += transaction[2]

    return successful_transactions, successful_volume

# Find cycles for circular rebalancing functionality.
def find_cycles(topology, source, destination, number_of_cycles):
    # returns at most number_of_cycles-many cycles of size at least 3 that end with channel
    # (source,destination) should be the last edge of the cycle
    # uses BFS discovery

    BFS_tree = nx.DiGraph()

    distance_k_nodes = {0: [destination]}
    visited = [source, destination]

    cycles = []
    distance_from_origin = 1
    max_depth_of_BFS = 10

    while len(cycles) <= number_of_cycles and distance_from_origin < max_depth_of_BFS:
        # find distance_from_origin nodes
        distance_k_nodes[distance_from_origin] = []

        for node in distance_k_nodes[distance_from_origin - 1]:
            for new_node in [e[1] for e in topology.out_edges(node) if e[1] not in visited]:
                # add to topology
                BFS_tree.add_edge(node, new_node)

                visited.append(new_node)

                # check for cycles
                if (new_node, source) in topology.edges:
                    new_cycle = nx.shortest_path(BFS_tree, destination, new_node) + [source]
                    cycles.append(new_cycle)

        distance_from_origin += 1

    return cycles[:number_of_cycles]

# Rebalance function for cilcular rebalancing.
def rebalance(topology, cycle, amount):
    # Rebalancing a cycle in the form: [v1, v2, ..., vk]

    channels_in_cycle = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

    # check if possible to rebalance based on: amount and willingness to rebalance
    topology = update_pc_states(topology)

    channel_states = {topology[u][v]['state'] for u, v in channels_in_cycle}
    channel_balances = {topology[u][v]['satoshis'] for u, v in channels_in_cycle}

    if 'not participating' in channel_states or amount > min(channel_balances):
        rebalancing_successful = False

        if 'not participating' in channel_states:
            print('not participating in channel_states')
        else:
            print('amount > min(channel_balances)')

    else:
        # execute rebalancing
        for channel in channels_in_cycle:
            src, dst = channel
            topology = channel_update(topology, src, dst, amount)

        rebalancing_successful = True

    return topology, rebalancing_successful

# Same functionality as in circular rebalancing.
def LN_local_rebalancing(topology, transaction_list, max_rebalancings_per_transaction, cycle_search_cutoff):
    # LN local rebalancing by cycle finding

    # init success counters
    successful_transactions = 0
    successful_volume = 0
    number_of_rebalanced_edges = 0
    rebalance_stats = {(i + 1): 0 for i in range(max_rebalancings_per_transaction)}

    # execute as many transactions as possible
    for transaction in transaction_list:
        src, dst, trx = transaction

        successful_transaction, failed_at_channel, topology = execute_transaction(topology, transaction)

        # if not successful, try to rebalance
        rebalancings = 0
        cycle_pool = []
        cycle_pool_empty = False

        while not successful_transaction and rebalancings < max_rebalancings_per_transaction and not cycle_pool_empty:
            # info of channel where the transaction failed
            x, y, fwd_amount = failed_at_channel

            # find cycles that include the reversed failed channel
            if not cycle_pool:
                cycle_pool = find_cycles(topology, y, x, cycle_search_cutoff)

                # if no cycle exists, break
                if not cycle_pool:
                    print(f'No cycles found! deg({x}) = {topology.degree[x]}')
                    break
                else:
                    cycle_pool_empty = False

            # rebalance
            cycle = cycle_pool.pop()

            # true after the last cycle was poped
            if not cycle_pool:
                cycle_pool_empty = True

            channels_in_cycle = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

            # if the initial balance is not enough to cover the forwarded amount, break
            if topology[x][y]['initial balance'] < fwd_amount:
                print("Transaction infeasible: initial balance doesn't suffice")
                break

            # rebalancing should restore the initial balance in channel (x,y)
            rebalancing_amount = topology[x][y]['initial balance'] - topology[x][y]['satoshis']

            # execute rebalancing if the first cycle has enough capacity
            topology, rebalancing_successful = rebalance(topology, cycle, rebalancing_amount)

            # check if the transaction can succeed
            if rebalancing_successful:
                print('Rebalancing succeeded!')
                number_of_rebalanced_edges += len(channels_in_cycle)
                rebalancings += 1
                rebalance_stats[rebalancings] += 1

                successful_transaction, new_failed_at_channel, topology = execute_transaction(topology, transaction)

                # if the transaction failed at a new channel, recompute the cycle pool
                if failed_at_channel != new_failed_at_channel:
                    cycle_pool = []
                    failed_at_channel = new_failed_at_channel
            else:
                print('Rebalancing failed!')

        # if the transaction was successful (with or without rebalancing), update the output
        if successful_transaction:
            successful_transactions += 1
            successful_volume += transaction[2]

            # print rebalancing statistics
    rebalancing_stats_str = ''
    for reb in rebalance_stats:
        rebalancing_stats_str += f'Channels rebalanced {reb} time(s): {rebalance_stats[reb]}. '
    print(f'rebalancing statistics: {rebalancing_stats_str}')

    return successful_transactions, successful_volume, number_of_rebalanced_edges

# Returns to G connected components of G in G.subgraph
def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)

# Decide whether a node is participating or not, currently "participating" by default
def pc_state(topology, channel):
    # no states
    return 'participating'

    # The code below was unreachable so I commented it out.
    # computes pc state with respect to rebalancing intention
    # s, d = channel
    # random_state = False
    #
    # if random_state:
    #     if random.random() < 0.3:
    #         state = "not participating"
    #     else:
    #         state = 'participating'
    #
    # else:
    #     # state according to channel balance
    #     ratio = topology[s][d]['satoshis']/topology[s][d]['initial balance']
    #
    #     if ratio <= 0.25:
    #         state = "depleted"
    #     elif ratio < 1:
    #         state = "participating"
    #     else:
    #         state = "not participating"
    #
    # return state

# Creating a PCN from a Lightning snapshot that is take the biggest subgraph and create parameters: initial balances, capacities, participating status
def prepare_LN_snapshot(average_transaction_amount):
    # prepare a PCN according to our model based on a snapshot of the Lightning Network

    machine = input('running on server or laptop?\n')

    if machine == 'laptop':
        # Read graph object in Python pickle format.
        snapshot = nx.read_gpickle('ln.gpickle')

        # Consider only components aka subgraphs with connected nodes e.g. for this case 48 components
        components = [c for c in connected_component_subgraphs(snapshot)]

        # Distribution of edges in each of these graphs
        edge_component_distribution = [components[i].number_of_edges() for i in range(len(components))]

        # Index of the component with the largest edges distribution
        largest_comp_index_by_edges = np.argmax(edge_component_distribution)

        # Take the largest component as a LN
        LN = components[largest_comp_index_by_edges]

    else:
        # read largest component json
        with open('LN-largest-cc.json', 'r') as handle:
            LN_dict = json.load(handle)

        LN = nx.Graph()
        LN.add_nodes_from(LN_dict['nodes'])
        LN.add_edges_from(LN_dict['channels'])

    topology = nx.DiGraph()
    for edge in LN.edges:
        s, d = edge

        # Capacity of each channel e.g. LN[6444][14775]['satoshis'] = 50000
        capacity = LN[s][d]['satoshis']

        # Consider an edge in two directions
        for pair in [(s, d), (d, s)]:
            # And make two pairs u and v from it
            u, v = pair

            # It means weight is the sum of base and relative fee multiplied with avg trans amount
            w = LN[u][v]['base_fee'] + LN[u][v]['relative_fee'] * average_transaction_amount
            topology.add_edge(u, v, weight=w)
            for key in LN[u][v]:
                if key == 'base_fee':
                    # Values conversion
                    topology[u][v][key] = LN[u][v][key] * 10 ** -3  # all values are converted to satoshis
                elif key == 'relative_fee':
                    # Values conversion
                    topology[u][v][key] = LN[u][v][
                                              key] * 10 ** -6  # all values are converted to satoshis
                else:
                    # key == 'satoshis'
                    # No conversion for capacity
                    topology[u][v][key] = LN[u][v][key]

        # Randomly initiate the channel initial balance
        topology[u][v]['satoshis'] = np.floor(topology[u][v]['satoshis'] / random.randint(2, 5))
        topology[u][v]['initial balance'] = topology[u][v]['satoshis']

        # Other side is the whole capacity minus random intitated from other side
        topology[v][u]['satoshis'] = capacity - topology[u][v]['satoshis']
        topology[v][u]['initial balance'] = topology[v][u]['satoshis']

        # compute initial rebalancing state
        topology[u][v]['state'] = pc_state(topology, (u, v))
        topology[v][u]['state'] = pc_state(topology, (v, u))

        # Thus, [satoshis] is capacity of a channel, and [initial balance] is amount of satoshis on one side or another
        # Also, [state] is an indicator if one of the nodes would like to rebalance this channel e.g. for his code always "participating"

    return topology

# Creating transactions on a particular topology created from LN snapshot
def transaction_lists_gen2(topology, number_of_unique_transactions, transaction_repetitions, tr_LB, tr_UB):
    # creates num_of_transaction_lists lists of random transactions
    list_of_transactions = []

    # Drawing sources and destinations. Creating direction.
    # draw sources it means we deal only with 100 randomly chosen unique transactions from all the graph?
    sources = random.choices(list(topology.nodes), k=number_of_unique_transactions)

    # draw destinations
    destinations = []
    for source in sources:
        destinations.append(random.choice([x for x in sources if x != source and (source, x) not in topology.edges]))

    # draw transaction amounts it means random values between two boundaries for 100 unique transactions
    transaction_amounts = random.choices(range(tr_LB, tr_UB), k=number_of_unique_transactions)

    transaction_base = []
    for i in range(number_of_unique_transactions):
        transaction_base.append((sources[i], destinations[i], transaction_amounts[i]))

    for number_of_repetitions in transaction_repetitions:
        new_list = transaction_base * number_of_repetitions
        random.shuffle(new_list)

        # Creation of 4 lists with different transaction scale {list:4}[[1000],[1200],[1400],[1600])
        list_of_transactions.append(new_list)

    return list_of_transactions

# TODO Hide & Seek skeleton Plugin usage,
#  Execute transaction until one of them is not possible, rebalance, continue carrying out transactions
def hide_and_seek(topology, transaction_list, global_rebalancing_threshold, num_of_conc_cores):

    successful_transactions = 0
    successful_volume = 0
    stacked_transactions = []
    # channels_to_rebalance = []
    channels_where_trx_failed = []
    trx_failed_twice = []

    # execute the transaction list (Is this for simulating the normal flow in LN?)
    for transaction in transaction_list:
        successful_transaction, failed_at_channel, topology = execute_transaction(topology, transaction)

        if successful_transaction:
            successful_transactions += 1
            successful_volume += transaction[2]

        # So execute rebalancing only on failed transaction channels?
        else:
            stacked_transactions.append(transaction)

            # The list of channels where transaction failed
            channels_where_trx_failed.append(failed_at_channel)

            # TODO Plugin usage rebalance every global_rebalancing_threshold stacked transactions,
            # so we start to rebalance only if threshold is reached
            if len(stacked_transactions) == global_rebalancing_threshold:
                # update pc states with respect to rebalancing aka check who is still willing to participate in Hide & Seek
                topology = update_pc_states(topology)

                # define rebalancing graph, as a non-linked copy of topology subgraph of topology
                rebalancing_graph = HS_rebalancing_graph(topology, channels_where_trx_failed)

                # compute flow updates
                flow_updates = LP_global_rebalancing(rebalancing_graph, num_of_conc_cores)

                number_of_rebalanced_channels = sum([1 for s, d, flow in flow_updates if flow != 0])
                total_flow_on_depleted_channels = sum([flow for s, d, flow in flow_updates])

                # apply flow through (u,v) equiv to adding flow to (v,u)
                for u, v, flow in flow_updates:
                    topology = channel_update(topology, u, v, flow)

                # try to execute the stacked transactions, put the failed ones in another list
                for transaction in stacked_transactions:
                    successful_transaction, failed_at_channel, topology = execute_transaction(topology, transaction)

                    if successful_transaction:
                        successful_transactions += 1
                        successful_volume += transaction[2]

                    else:
                        trx_failed_twice.append(transaction)

                post(
                    f'number of rebalanced channels: {number_of_rebalanced_channels}, total flow on participating channels: {total_flow_on_depleted_channels}')

                # clear stacked transactions to run other transation
                stacked_transactions = []

    return successful_transactions, successful_volume, len(trx_failed_twice), len(trx_failed_twice)

# TODO be clarified PLUGIN
def execute_transaction(topology, transaction):
    # execute a single transaction
    source, destination, amount = transaction
    successful_transaction = True
    failed_at_channel = False

    # Сompute shortest path according to weights with a help of networkx library
    shortest_path = nx.shortest_path(topology, source, destination, weight='weight')
    shortest_path_edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]

    # I don't get why should here .reverse be used.
    shortest_path_edges.reverse()

    # compute routing fees backwards and check if the capacities suffice
    stacked_payments = []
    current_edge = shortest_path_edges.pop()
    hop, nexthop = current_edge
    current_amount = amount
    current_fee = topology[hop][nexthop]['base_fee'] + topology[hop][nexthop]['relative_fee'] * current_amount

    # Execute only if transaction amount is smaller than overall capacity
    if topology[hop][nexthop]['satoshis'] > current_amount:
        # if the last edge in the path has enough capacity, proceed to checking the previous ones
        stacked_payments.append((hop, nexthop, current_amount))

        # repeat backwards for the remaining edges
        while shortest_path_edges:

            hop, nexthop = shortest_path_edges.pop()
            current_amount += current_fee
            current_fee = topology[hop][nexthop]['base_fee'] + topology[hop][nexthop]['relative_fee'] * current_amount

            if topology[hop][nexthop]['satoshis'] > current_amount:
                stacked_payments.append((hop, nexthop, current_amount))
            else:
                successful_transaction = False
                failed_at_channel = (hop, nexthop, amount)
                break
    else:
        successful_transaction = False
        failed_at_channel = (hop, nexthop, amount)

    # if the transaction can be carried out, execute the transaction
    if successful_transaction:
        while stacked_payments:
            (src, dst, trx) = stacked_payments.pop()
            topology = channel_update(topology, src, dst, trx)

    return successful_transaction, failed_at_channel, topology

# TODO Plugin usage Change the channel states, PLUGIN as RPC call to change states in implementation
def channel_update(topology, from_node, to_node, amount):
    topology[from_node][to_node]['satoshis'] -= amount
    topology[to_node][from_node]['satoshis'] += amount

    return topology

# TODO PLUGIN usage
def update_pc_states(topology):
    # updates pc state with respect to rebalancing intention
    for channel in topology.edges:
        u, v = channel
        topology[u][v]['state'] = pc_state(topology, channel)

    return topology

# TODO Plugin usage
def HS_rebalancing_graph(topology, channels_where_trx_failed):
    # constructs rebalancing graph for Hide & Seek
    rebalancing_graph = nx.DiGraph()

    # rebalancing amount (?)
    # if max, then assume MPC
    max_flow = max(
        {topology[x][y]['initial balance'] - topology[x][y]['satoshis'] for x, y, z in channels_where_trx_failed})
    print(f'm(u,v) variables set to {max_flow}')

    # setup channels and maximum flow_bound
    rebalancing_graph_edges = [(u, v) for u, v in topology.edges if topology[u][v]['state'] != 'not participating']
    rebalancing_graph.add_edges_from(rebalancing_graph_edges, flow_bound=max_flow)

    # if a transaction failed at (src,dst), set 0 flow to this direction and let max_flow in the reverse
    for src, dst, val in channels_where_trx_failed:
        rebalancing_graph[src][dst]['flow_bound'] = 0
        rebalancing_graph[dst][src]['objective function coefficient'] = 1

        # set objective function coefficients
    for u, v in topology.edges:
        if topology[u][v]['state'] == 'depleted':
            rebalancing_graph[u][v]['objective function coefficient'] = 1
            rebalancing_graph[u][v]['flow_bound'] = 0

        # adjust flow bounds if they exceed a preset limit
        if rebalancing_graph[u][v]['flow_bound'] > topology[u][v]['satoshis'] / 2:
            rebalancing_graph[u][v]['flow_bound'] = topology[u][v]['satoshis'] / 2

    # rebalancing_channels = {(u,v) for u,v in topology.edges if topology[u][v]['state'] != 'not participating'}
    # rebalancing_channels -= set(channels_where_trx_failed)

    # # add channels and their intended rebalancing amounts.
    # rebalancing_graph = topology.edge_subgraph(rebalancing_channels).copy()

    # for channel in rebalancing_graph.edges:
    #     u,v = channel
    #     rebalancing_graph[u][v]['flow_bound'] = topology[v][u]['initial balance'] - topology[v][u]['satoshis']

    # for channel in channels_where_trx_failed:
    #     u,v = channel
    #     rebalancing_graph.add_edge(u,v)

    #     # optimally, the amount of flow through (u,v) should be the satoshis needed such that (v,u),
    #     # the channel to be refunded, returns to its initial balance
    #     rebalancing_graph[u][v]['max flow'] = topology[v][u]['initial balance'] - topology[v][u]['satoshis']

    return rebalancing_graph

# TODO Plugin usage
def LP_global_rebalancing(rebalancing_graph, num_of_conc_cores):
    # gloabal rebalancing LP
    # same structure as https://www.gurobi.com/documentation/9.1/quickstart_mac/cs_example_matrix1_py.html

    last_time = time.time()

    try:
        n = rebalancing_graph.number_of_nodes()
        m = rebalancing_graph.number_of_edges()
        global list_of_nodes
        global list_of_edges
        list_of_nodes = list(rebalancing_graph.nodes)
        list_of_edges = list(rebalancing_graph.edges)

        # Create a new model
        model = gp.Model("rebalancing-LP")

        # Create variables
        x = model.addMVar(shape=m, vtype=GRB.CONTINUOUS, name="x")

        # Set objective
        obj = np.zeros(m, dtype=float)

        # only channels where transactions failed contribute to the objective function
        for edge_index in range(m):
            u, v = list_of_edges[edge_index]
            if 'objective function coefficient' in rebalancing_graph[u][v]:
                obj[edge_index] = rebalancing_graph[u][v]['objective function coefficient']

        model.setObjective(obj @ x, GRB.MAXIMIZE)

        ## adding constraints
        # init
        data = []
        row = []
        col = []
        rhs = np.zeros(2 * m + 2 * n)

        # constraint 1: respecting capacities: 0 <= f(u,v) <= m(u,v)
        # I.e. -f(u,v) <= 0 and f(u,v) <= m(u,v)
        # sequential code
        # for edge_index in range(m):
        #     u,v = list_of_edges[edge_index]

        #     # -f(u,v) <= 0
        #     append_to_A(-1, edge_index, edge_index, data, row, col)
        #     # bound is zero, so no need to edit rhs
        #     # rhs[edge_index] = 0

        #     # f(u,v) <= m(u,v)
        #     append_to_A(1, m + edge_index, edge_index, data, row, col)
        #     rhs[m + edge_index] = rebalancing_graph[u][v]['flow_bound']

        # parallel code
        # 1a: -f(u,v) <= 0
        inputs_1a = range(m)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_of_conc_cores) as executor:
            results_1a = executor.map(compute_costraint_1a, inputs_1a)  # results is a generator object

        # 1b: f(u,v) <= m(u,v)
        inputs_1b = [(m, edge_index) for edge_index in range(m)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_of_conc_cores) as executor:
            results_1b = executor.map(compute_costraint_1b, inputs_1b)  # results is a generator object

        all_entries = [result for result in results_1a] + [result for result in results_1b]  # unpack generator

        # set bound vector
        for edge_index in range(m):
            u, v = list_of_edges[edge_index]
            rhs[m + edge_index] = rebalancing_graph[u][v]['flow_bound']

        print(f'done with constraint 1 in {duration_since(last_time)}')
        last_time = time.time()

        # constraint 2: flow conservation: sum of in flows = some of out flows
        # ineq 2a: \sum_{out edges} f(u,v) - \sum_{in edges} f(v,u) <= 0
        # ineq 2b: \sum_{in edges} f(v,u) - \sum_{out edges} f(u,v) <= 0
        # bounds (rhs vector) are already set to zero from initialization
        # sequeantial code
        # for i in range(n):
        #     # all bounds are zero, thus no need to edit rhs

        #     u = list_of_nodes[i]

        #     for edge in rebalancing_graph.out_edges(u):
        #         edge_index = list_of_edges.index(edge)

        #         # ineq 2a: \sum_{out edges} f(u,v)
        #         append_to_A(1, 2*m + i, edge_index, data, row, col)

        #         # ineq 2b: - \sum_{out edges} f(u,v)
        #         append_to_A(-1, 2*m + n + i, edge_index, data, row, col)

        #     for edge in rebalancing_graph.in_edges(u):
        #         edge_index = list_of_edges.index(edge)

        #         # ineq 2a: - \sum_{in edges} f(v,u)
        #         append_to_A(-1, 2*m + i, edge_index, data, row, col)

        #         # ineq 2b: \sum_{in edges} f(v,u)
        #         append_to_A(1, 2*m + n + i, edge_index, data, row, col)

        # parallel code
        inputs_2 = [(n, m, node_index) for node_index in range(n)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_of_conc_cores) as executor:
            results_1b = executor.map(compute_costraint_2, inputs_2)  # results is a generator object

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

# TODO Plugin usage (not sure) LP usage (so not needed in par representation?)
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

# TODO Plugin usage, constraints for LP.
def compute_costraint_1a(edge_index):
    # -f(u,v) <= 0
    return (-1, edge_index, edge_index)

# TODO Plugin usage, constraints for LP.
def compute_costraint_1b(args):
    # f(u,v) <= m(u,v)
    m, edge_index = args
    return (1, m + edge_index, edge_index)

# TODO Plugin usage, constraints for LP.
def compute_costraint_2(args):
    n, m, node_index = args

    u = list_of_nodes[node_index]

    tuples = []

    for edge in topology.out_edges(u):
        edge_index = list_of_edges.index(edge)

        # ineq 2a: \sum_{out edges} f(u,v)
        tuples.append((1, 2 * m + node_index, edge_index))

        # ineq 2b: - \sum_{out edges} f(u,v)
        tuples.append((-1, 2 * m + n + node_index, edge_index))

        for edge in topology.in_edges(u):
            edge_index = list_of_edges.index(edge)

        # ineq 2a: - \sum_{in edges} f(v,u)
        tuples.append((-1, 2 * m + node_index, edge_index))

        # ineq 2b: \sum_{in edges} f(v,u)
        tuples.append((1, 2 * m + n + node_index, edge_index))

        return tuples


global rebalancing_graph

# record start time
start_time = time.time()
last_time = start_time

# bottom 11 initial capacities of existing snapshot: [220.0, 275.0, 300.0, 400.0, 500.0, 550.0, 666.0, 825.0, 880.0, 1000.0, 1200.0]
# Low and upper bounds of all capacities
tr_LB = 1100  # ~$0.5 this is the minimum capacity
tr_UB = 2 * 5000  # ~$2.2*3. max capacity: 500000000
# Floor of the average amount e.g. for this code is (10000 - 1100)/2 -> floor of 4450
average_transaction_amount = np.floor((tr_UB - tr_LB) / 2)

# Creating topology from snapshot ça te dit a graph with initial balances, capacities, weights and fees, transactions and states.
topology = prepare_LN_snapshot(average_transaction_amount)
initial_topology = copy.deepcopy(topology)

# set up the input and experiment parameters
number_of_unique_transactions = 100
transaction_repetitions = [10, 12, 14, 16]
# transaction_repetitions = [2, 4, 6, 8, 10]
# Number of transaction list is the length of general amount of repetitions = 4
number_of_transaction_lists = len(transaction_repetitions)

# LN local rebalancing input
cycle_search_cutoff = 15
max_rebalancings_per_transaction = 1

# Hide & Seek input
global_rebalancing_threshold = 50
num_of_conc_cores = 3

naming_suffix = f'-reb{global_rebalancing_threshold}-min{min(transaction_repetitions)}-max{max(transaction_repetitions)}'
results_file_name = f'rebalancing-results{naming_suffix}.json'
compute_new_results = True

# set up log
log_file_name = f'rebalancing-results{naming_suffix}.txt'
f = open(log_file_name, 'w')
f.close()
post = lambda text: write_and_print(log_file_name, text)
post(f'logging results for {global_rebalancing_threshold} stacked transaction for H&S')

# create transaction lists
transaction_lists = transaction_lists_gen2(topology, number_of_unique_transactions, transaction_repetitions, tr_LB, tr_UB)

# init results dict
results = {i: {'no rebalancing': {}, 'LN local': {}, 'Hide & Seek': {}} for i in range(number_of_transaction_lists)}
methods_used = ['no rebalancing', 'LN local', 'Hide & Seek']


if compute_new_results:
    trx_lists_ids = range(number_of_transaction_lists)
else:
    trx_lists_ids = [str(i) for i in range(number_of_transaction_lists)]

if compute_new_results:
    # compute the results for each method and list of transactions
    for i in range(number_of_transaction_lists):
        total_num_trans = len(transaction_lists[i])
        total_volume = sum([transaction[2] for transaction in transaction_lists[i]])

        for method in methods_used:
            if method == 'no rebalancing':
                succ_trans, succ_vol = no_rebalancing(topology, transaction_lists[i])
                print(
                    f'No rebalancing: for {transaction_repetitions[i]} repetitions, success ratio = {succ_trans / total_num_trans} and success volume ratio = {succ_vol / total_volume}')

            elif method == 'LN local':
                succ_trans, succ_vol, number_of_rebalanced_edges = LN_local_rebalancing(topology, transaction_lists[i],
                                                                                        max_rebalancings_per_transaction,
                                                                                        cycle_search_cutoff)
                print(
                    f'LN local rebalancing: for {transaction_repetitions[i]} repetitions, success ratio = {succ_trans / total_num_trans}, success volume ratio = {succ_vol / total_volume}, and number_of_rebalanced_edges = {number_of_rebalanced_edges}')

            else:
                # There was a parameter num_of_conc_cores missing so I've just added it
                # hide & seek
                succ_trans, succ_vol, number_of_rebalanced_edges, trx_failed_twice = hide_and_seek(topology,
                                                                                                   transaction_lists[i],
                                                                                                   global_rebalancing_threshold,
                                                                                                   num_of_conc_cores)
                print(
                    f'Hide & Seek: for {transaction_repetitions[i]} repetitions, success ratio = {succ_trans / total_num_trans}, success volume ratio = {succ_vol / total_volume}, and number of transactions that failed twice = {trx_failed_twice}')

            # store results for plotting
            results[i][method]['successful_transactions'] = succ_trans / total_num_trans
            results[i][method]['successful_volume'] = succ_vol / total_volume

            results[i][method]['successful_transactions_annotation'] = int((succ_trans / total_num_trans) * 100) / 100
            results[i][method]['successful_volume_annotation'] = int((succ_vol / total_volume) * 100) / 100

            topology = copy.deepcopy(initial_topology)

            # log results
            post(
                f'Duration of {method} for {transaction_repetitions[i]} transaction repetitions: {duration_since(last_time)}')
            last_time = time.time()

        post(' ')

    # save the results
    with open(results_file_name, 'w') as handle:
        json.dump(results, handle)

else:
    with open(results_file_name, 'r') as handle:
        results = json.load(handle)





## plot the results
# success ratio plot
for method in methods_used:
    success_tr_ratios = [results[i][method]['successful_transactions'] for i in trx_lists_ids]
    # success_vol_ratios = [results[i][method]['successful_volume'] for i in range(number_of_transaction_lists)]

    success_tr_ratios_annotation = [results[i][method]['successful_transactions_annotation'] for i in trx_lists_ids]
    # success_vol_ratios_annotation = [results[i][method]['successful_volume_annotation'] for i in range(number_of_transaction_lists)]

    plt.plot(transaction_repetitions, success_tr_ratios, '--o')

    for i in range(number_of_transaction_lists):
        plt.annotate(success_tr_ratios_annotation[i], (transaction_repetitions[i], success_tr_ratios[i]))

plt.xlabel("number of repetitions of individual transactions")
plt.ylabel("transaction success ratio")
# plt.xticks(rotation=80)
# plt.title("Transaction success ratio")
plt.grid()
plt.legend(methods_used)  # , loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
plt.savefig(f'tr-success-ratio{naming_suffix}.pdf', bbox_inches="tight")  # , bbox_inches="tight"
plt.show()

# success volume plot
# plot the results
for method in methods_used:
    # success_tr_ratios = [results[i][method]['successful_transactions'] for i in range(number_of_transaction_lists)]
    success_vol_ratios = [results[i][method]['successful_volume'] for i in trx_lists_ids]

    # success_tr_ratios_annotation = [results[i][method]['successful_transactions_annotation'] for i in range(number_of_transaction_lists)]
    success_vol_ratios_annotation = [results[i][method]['successful_volume_annotation'] for i in trx_lists_ids]

    plt.plot(transaction_repetitions, success_vol_ratios, '--o')

    for i in range(number_of_transaction_lists):
        plt.annotate(success_vol_ratios_annotation[i], (transaction_repetitions[i], success_vol_ratios_annotation[i]))

plt.xlabel("number of repetitions of individual transactions")
plt.ylabel("volume success ratio")
# plt.xticks(rotation=80)
plt.title(f"Hide & Seek rebalancing at {global_rebalancing_threshold} stacked transactions")
plt.grid()
plt.legend(methods_used)
plt.savefig(f'vol-success-ratio{naming_suffix}.pdf', bbox_inches="tight")
plt.show()

post(f'Total duration: {duration_since(start_time)}')
