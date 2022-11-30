import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt


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


def LP_global_rebalancing(rebalancing_graph) -> list:
    # Step 5: Executes LP on that graph (need a license be deployed)
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
            if 'objective_function_coefficient' in rebalancing_graph[u][v]:
                obj[edge_index] = rebalancing_graph[u][v]['objective_function_coefficient']


        model.setObjective(obj @ x, GRB.MAXIMIZE)

        data = []
        row = []
        col = []
        rhs = np.zeros(2 * m + 2 * n)

        # constraint 1: respecting capacities: 0 <= f(u,v) <= m(u,v)
        # I.e. -f(u,v) <= 0 and f(u,v) <= m(u,v)
        for edge_index in range(m):
            u, v = list_of_edges[edge_index]

            # -f(u,v) <= 0
            append_to_A(-1, edge_index, edge_index, data, row, col)

            # f(u,v) <= m(u,v)
            append_to_A(1, m + edge_index, edge_index, data, row, col)
            rhs[m + edge_index] = rebalancing_graph[u][v]['flow_bound']

        for edge_index in range(m):
            u, v = list_of_edges[edge_index]
            rhs[m + edge_index] = rebalancing_graph[u][v]['flow_bound']

        print(f'done with constraint 1')

        # constraint 2: flow conservation: sum of in flows = some of out flows
        # ineq 2a: \sum_{out edges} f(u,v) - \sum_{in edges} f(v,u) <= 0
        # ineq 2b: \sum_{in edges} f(v,u) - \sum_{out edges} f(u,v) <= 0
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

        # Add constraints and optimize model
        model.addConstr(A @ x <= rhs, name="matrix form constraints")
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

        return balance_updates

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')


def cycle_decomposition(balance_updates, rebalancing_graph) -> list:
    # Step 6: Cycle decomposition on MPC delegate
    # Ask maybe we should clean balance_updates before?
    cycle_flows = [[]]

    # Clean balances updates from zero ones and create dictionary
    active_edges = list(filter(lambda edge: edge[2] != 0, balance_updates))
    active_edges_dictionary = dict([((a, b), c) for a, b, c in active_edges])

    i = 0
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
        min_flow = min(dict(circulation_graph_weighted.edges).items(), key=lambda x: x[1]['weight'])
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


def main():
    # The simplest graph
    triangle = nx.DiGraph()
    triangle.add_nodes_from(['Alice', 'Bob', 'Carol'])

    # Alice --> Bob and Alice <-- Bob
    triangle.add_edge('Alice', 'Bob')
    triangle.add_edge('Bob', 'Alice')

    triangle['Alice']['Bob']['initial_balance'] = 20
    triangle['Bob']['Alice']['initial_balance'] = 10

    triangle['Alice']['Bob']['satoshis'] = 30
    triangle['Bob']['Alice']['satoshis'] = 30

    triangle['Alice']['Bob']['objective_function_coefficient'] = 1
    triangle['Alice']['Bob']['flow_bound'] = 5
    triangle['Bob']['Alice']['flow_bound'] = 0

    # Bob --> Carol and Bob <-- Carol
    triangle.add_edge('Bob', 'Carol')
    triangle.add_edge('Carol', 'Bob')

    triangle['Carol']['Bob']['initial_balance'] = 4
    triangle['Bob']['Carol']['initial_balance'] = 10

    triangle['Carol']['Bob']['satoshis'] = 14
    triangle['Bob']['Carol']['satoshis'] = 14

    triangle['Bob']['Carol']['objective_function_coefficient'] = 1
    triangle['Carol']['Bob']['flow_bound'] = 0
    triangle['Bob']['Carol']['flow_bound'] = 3

    # Carol --> Alice and Carol <-- Alice
    triangle.add_edge('Carol', 'Alice')
    triangle.add_edge('Alice', 'Carol')

    triangle['Carol']['Alice']['initial_balance'] = 40
    triangle['Alice']['Carol']['initial_balance'] = 20

    triangle['Carol']['Alice']['satoshis'] = 60
    triangle['Alice']['Carol']['satoshis'] = 60

    triangle['Carol']['Alice']['objective_function_coefficient'] = 1
    triangle['Carol']['Alice']['flow_bound'] = 10
    triangle['Alice']['Carol']['flow_bound'] = 0

    lp_outcome = LP_global_rebalancing(triangle)
    print(lp_outcome)
    cycle_decomposition(lp_outcome)
    print(cycle_decomposition())


if __name__ == "__main__":
    main()
