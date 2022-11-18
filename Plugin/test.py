from itertools import repeat
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt


# Step 5: Executes LP on that graph (need a license be deployed)
def LP_global_rebalancing(rebalancing_graph):
    try:
        n = rebalancing_graph.number_of_nodes()
        m = rebalancing_graph.number_of_edges()
        list_of_nodes = list(rebalancing_graph.nodes)
        list_of_edges = list(rebalancing_graph.edges)

        # Create a new model, variables and set an objective
        model = gp.Model("rebalancing-LP")
        x = model.addMVar(shape=m, vtype=GRB.CONTINUOUS, name="x")
        obj = np.zeros(m, dtype=float)

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
        for edge_index in range(m):
            u, v = list_of_edges[edge_index]

            # -f(u,v) <= 0
            append_to_A(-1, edge_index, edge_index, data, row, col)
            # bound is zero, so no need to edit rhs
            # rhs[edge_index] = 0

            # f(u,v) <= m(u,v)
            append_to_A(1, m + edge_index, edge_index, data, row, col)
            rhs[m + edge_index] = rebalancing_graph[u][v]['flow_bound']

        # set bound vector
        for edge_index in range(m):
            u, v = list_of_edges[edge_index]
            rhs[m + edge_index] = rebalancing_graph[u][v]['flow_bound']

        print(f'done with constraint 1')

        # constraint 2: flow conservation: sum of in flows = some of out flows
        # ineq 2a: \sum_{out edges} f(u,v) - \sum_{in edges} f(v,u) <= 0
        # ineq 2b: \sum_{in edges} f(v,u) - \sum_{out edges} f(u,v) <= 0
        # bounds (rhs vector) are already set to zero from initialization
        for i in range(n):
            # all bounds are zero, thus no need to edit rhs

            u = list_of_nodes[i]

            for edge in rebalancing_graph.out_edges(u):
                edge_index = list_of_edges.index(edge)

                # ineq 2a: \sum_{out edges} f(u,v)
                append_to_A(1, 2 * m + i, edge_index, data, row, col)

                # ineq 2b: - \sum_{out edges} f(u,v)
                append_to_A(-1, 2 * m + n + i, edge_index, data, row, col)

            for edge in rebalancing_graph.in_edges(u):
                edge_index = list_of_edges.index(edge)

                # ineq 2a: - \sum_{in edges} f(v,u)
                append_to_A(-1, 2 * m + i, edge_index, data, row, col)

                # ineq 2b: \sum_{in edges} f(v,u)
                append_to_A(1, 2 * m + n + i, edge_index, data, row, col)

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


# Step 6: Cycle decomposition on MPC delegate
# Ask maybe we should clean balance_updates before?
def cycle_decomposition(balance_updates, rebalancing_graph):
    cycle_flows = [[]]


    # Clean balances updates from zero ones and create dictionary
    active_edges = list(filter(lambda edge: edge[2] != 0, balance_updates))
    active_edges_dictionary = dict([((a, b), c) for a, b, c in active_edges])

    i = 0
    # Active edges dictionary loaded correctly
    while len(active_edges_dictionary) > 0:

        # Weighted circulation graph and start counter for circles
        circulation_graph_weighted = nx.DiGraph()

        # Update graph from dictionary for cycle search
        for e, w in active_edges_dictionary.items():
            circulation_graph_weighted.add_edge(e[0], e[1], weight=w)

        # Find a cycle using DFS
        cycle = nx.find_cycle(circulation_graph_weighted)

        # Add weights to cycle
        for e in range(len(cycle)):
            weight = active_edges_dictionary.get(cycle[e])
            cycle[e] = (cycle[e][0], cycle[e][1], weight)

        # Create a weighted graph from weighted cycle
        cycle_graph = nx.DiGraph()
        cycle_graph.add_weighted_edges_from(cycle)

        # Find a minimum flow in the circulation graph
        min_flow = min(dict(cycle_graph.edges).items(), key=lambda x: x[1]['weight'])
        smallest_weight = min_flow[1]['weight']

        # Create a cycle
        for edge in cycle_graph.edges:
            new_balance_update = (edge[0], edge[1], smallest_weight)
            cycle_flows[i].append(new_balance_update)

            active_edges_dictionary[edge] = active_edges_dictionary.get(edge) - new_balance_update[2]

            if active_edges_dictionary[edge] == 0:
                active_edges_dictionary.pop(edge)

        i += 1
        cycle_flows.append([])

    cycle_flows.pop()

    return cycle_flows



def htlc_creation_for_cycles(cycles):

    for cycle in cycles:
        # Assumption: we are executing all the cycle from the node initiator
        # In the paper it happens randomly between all of the nodes in the cycles
        invoice = plugin.rpc.invoice(msatoshi, label, description, retry_for + 60)
        payment_hash = invoice['payment_hash']

        # The requirement for payment_secret coincided with its addition to the invoice output.
        payment_secret = invoice.get('payment_secret')
        # timelock tc ←− len(c)
        # timelock tc ←− len(c)
        # uc chooses random secret rc and creates hash hc = H(rc)
        # for ec = (u, v) ∈ c starting from uc do
        # u creates HTLC(u, v, wc, hc, tc)
        # decrement tc by 1

    return 1


def plot_graph_with_capacities(graph):
    # Plot with capacities
    plot_graph_with_capacities()
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True)
    edge_labels = dict([((x, y), e['satoshis'])
                        for x, y, e in graph.edges(data=True)])

    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.show()


def plot_graph_with_initial_balances(graph):
    # Plot with initial balances
    # colors = nx.get_edge_attributes(graph, 'color').values()
    # nx.draw_networkx(graph, pos, with_labels=True, connectionstyle='arc3, rad = 0.1', edge_color=colors)
    cmap = plt.cm.viridis(np.linspace(0, 1, graph.number_of_edges()))
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos, with_labels=True, connectionstyle='arc3, rad = 0.1')



    # nx.draw_networkx_edges(graph, pos, edge_color=cmap)
    edge_labels_bidirectional = dict([((u, v,), d['initial_balance'])
                         for u, v, d in graph.edges(data=True)])
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels_bidirectional,
                                 label_pos=0.3, font_size=7)
    nx.draw_networkx_edges(graph, pos, edge_color=cmap)
    [nx.draw_networkx_edge_labels(graph, pos, edge_labels={e: i}, font_color=cmap[i]) for i, e in enumerate(graph.edges())]
    plt.show()


def main():
    # The simplest graph
    #      (20) Alice (20)
    #           ^   ^
    #     (40) /      \  (10)
    #   Carol <--------> Bob
    #     (10)          (40)

    graph = nx.DiGraph()
    graph.add_nodes_from(['Alice', 'Bob', 'Carol', 'Emma', 'Dave'])

    # Alice --> Bob and Alice <-- Bob
    graph.add_edge('Alice', 'Bob', color='r')
    graph.add_edge('Bob', 'Alice', color='g')
    # both sides
    graph['Alice']['Bob']['initial_balance'] = 20
    graph['Bob']['Alice']['initial_balance'] = 10
    # capacities
    graph['Alice']['Bob']['satoshis'] = 30
    graph['Bob']['Alice']['satoshis'] = 30
    # flows
    graph['Alice']['Bob']['flow_bound'] = 5
    graph['Bob']['Alice']['flow_bound'] = 0
    # objective
    graph['Alice']['Bob']['objective function coefficient'] = 1

    # Bob --> Carol and Bob <-- Carol
    graph.add_edge('Bob', 'Carol', color='g')
    graph.add_edge('Carol', 'Bob', color='b')
    # both sides
    graph['Bob']['Carol']['initial_balance'] = 40
    graph['Carol']['Bob']['initial_balance'] = 10
    # capacities
    graph['Bob']['Carol']['satoshis'] = 50
    graph['Carol']['Bob']['satoshis'] = 50
    # flows
    graph['Bob']['Carol']['flow_bound'] = 15
    graph['Carol']['Bob']['flow_bound'] = 0
    # objective
    graph['Alice']['Bob']['objective function coefficient'] = 1

    # Carol --> Alice and Carol <-- Alice
    graph.add_edge('Carol', 'Alice', color='b')
    graph.add_edge('Alice', 'Carol', color='r')
    # both sides
    graph['Carol']['Alice']['initial_balance'] = 40
    graph['Alice']['Carol']['initial_balance'] = 20
    # capacities
    graph['Carol']['Alice']['satoshis'] = 60
    graph['Alice']['Carol']['satoshis'] = 60
    # objective
    graph['Carol']['Alice']['objective function coefficient'] = 1
    # flows
    graph['Carol']['Alice']['flow_bound'] = 10
    graph['Alice']['Carol']['flow_bound'] = 0

    # Alice --> Dave and Alice <-- Dave
    graph.add_edge('Alice', 'Dave', color='r')
    graph.add_edge('Dave', 'Alice', color='y')
    # both sides
    graph['Alice']['Dave']['initial_balance'] = 70
    graph['Dave']['Alice']['initial_balance'] = 30
    # capacities
    graph['Alice']['Dave']['satoshis'] = 100
    graph['Dave']['Alice']['satoshis'] = 100
    # flows
    graph['Alice']['Dave']['flow_bound'] = 20
    graph['Dave']['Alice']['flow_bound'] = 0
    # objective
    graph['Alice']['Dave']['objective function coefficient'] = 1

    # Alice --> Emma and Alice <-- Emma
    graph.add_edge('Alice', 'Emma', color='r')
    graph.add_edge('Emma', 'Alice', color='m')
    # both sides
    graph['Alice']['Emma']['initial_balance'] = 20
    graph['Emma']['Alice']['initial_balance'] = 20
    # capacities
    graph['Alice']['Emma']['satoshis'] = 40
    graph['Emma']['Alice']['satoshis'] = 40
    # flows
    graph['Alice']['Emma']['flow_bound'] = 0
    graph['Emma']['Alice']['flow_bound'] = 0

    # Bob --> Dave and Bob <-- Dave
    graph.add_edge('Bob', 'Dave', color='g')
    graph.add_edge('Dave', 'Bob', color='y')
    # both sides
    graph['Bob']['Dave']['initial_balance'] = 20
    graph['Dave']['Bob']['initial_balance'] = 70
    # capacities
    graph['Bob']['Dave']['satoshis'] = 90
    graph['Dave']['Bob']['satoshis'] = 90
    # flows
    graph['Bob']['Dave']['flow_bound'] = 0
    graph['Dave']['Bob']['flow_bound'] = 25
    # objective
    graph['Dave']['Bob']['objective function coefficient'] = 1

    # Bob --> Emma and Bob <-- Emma
    graph.add_edge('Bob', 'Emma', color='g')
    graph.add_edge('Emma', 'Bob', color='m')
    # both sides
    graph['Bob']['Emma']['initial_balance'] = 50
    graph['Emma']['Bob']['initial_balance'] = 60
    # capacities
    graph['Bob']['Emma']['satoshis'] = 110
    graph['Emma']['Bob']['satoshis'] = 110
    # flows
    graph['Bob']['Emma']['flow_bound'] = 0
    graph['Emma']['Bob']['flow_bound'] = 5
    # objective
    graph['Emma']['Bob']['objective function coefficient'] = 1

    # Carol --> Dave and Carol <-- Dave
    graph.add_edge('Carol', 'Dave', color='b')
    graph.add_edge('Dave', 'Carol', color='y')
    # both sides
    graph['Carol']['Dave']['initial_balance'] = 80
    graph['Dave']['Carol']['initial_balance'] = 20
    # capacities
    graph['Carol']['Dave']['satoshis'] = 100
    graph['Dave']['Carol']['satoshis'] = 100
    # objective
    graph['Carol']['Dave']['objective function coefficient'] = 1
    # flows
    graph['Carol']['Dave']['flow_bound'] = 30
    graph['Dave']['Carol']['flow_bound'] = 0

    # Carol --> Emma and Carol <-- Emma
    graph.add_edge('Carol', 'Emma', color='b')
    graph.add_edge('Emma', 'Carol', color='m')
    # both sides
    graph['Carol']['Emma']['initial_balance'] = 10
    graph['Emma']['Carol']['initial_balance'] = 90
    # capacities
    graph['Carol']['Emma']['satoshis'] = 100
    graph['Emma']['Carol']['satoshis'] = 100
    # objective
    graph['Emma']['Carol']['objective function coefficient'] = 1
    # flows
    graph['Carol']['Emma']['flow_bound'] = 0
    graph['Emma']['Carol']['flow_bound'] = 40

    # Plotting
    # plot_graph_with_capacities(graph)
    # plot_graph_with_initial_balances(graph)

    balance_updates = LP_global_rebalancing(graph)
    list_of_cycles = cycle_decomposition(balance_updates, graph)
    htlc_creation_for_cycles(list_of_cycles)


if __name__ == "__main__":
    main()
