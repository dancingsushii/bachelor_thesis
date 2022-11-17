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



def foo(somelist):
    return {x[2]:x for x in somelist}


# Step 6: Cycle decomposition on MPC delegate
def cycle_decomposition(balance_updates, rebalancing_graph):
    cycle_flows = []

    # [('Alice', 'Dave', 10), ('Bob', 'Carol', 15), ('Carol', 'Alice', 10), ('Carol', 'Dave', 5), ('Dave', 'Bob', 15)]
    active_edges = list(filter(lambda edge: edge[2] != 0, balance_updates))
    # {('Alice', 'Dave'): 10, ('Bob', 'Carol'): 15, ('Carol', 'Alice'): 10, ('Carol', 'Dave'): 5, ('Dave', 'Bob'): 15}
    active_edges_dictionary = dict([((a, b), c) for a, b, c in active_edges])

    # Weighted circulation graph
    circulation_graph_weighted = nx.DiGraph()
    i = 1

    while len(active_edges_dictionary) > 0:

        # circulation_graph_weighted.add_edges_from(active_edges_dictionary)
        # circulation_graph_weighted.add_nodes_from(active_edges_dictionary.keys())

        for e, w in active_edges_dictionary.items():
            circulation_graph_weighted.add_edge(e[0], e[1], weight=w)


        cycle = nx.find_cycle(circulation_graph_weighted)
        min_flow = min(dict(circulation_graph_weighted.edges).items(), key=lambda x: x[1]['weight'])
        smallest_weight = min_flow[1]['weight']

        for edge in cycle:
            edge = (edge[0], edge[1], circulation_graph_weighted[edge[0]][edge[1]]['weight'])
            new_balance_update = (edge[0], edge[1], smallest_weight)
            cycle_flows.append(new_balance_update)
            edge = (edge[0], edge[1], edge[2] - new_balance_update[2])

            # active_edges change the edge value

            # active_edges[edge[0]][edge[1]]

            if edge[2] == 0:
                active_edges_dictionary.pop(edge)
                # circulation_graph_weighted.remove(edge)

        i += 1

    return cycle_flows


def find_all_cycles(G, source=None, cycle_length_limit=None):
    """forked from networkx dfs_edges function. Assumes nodes are integers, or at least
    types which work with min() and > ."""
    if source is None:
        # produce edges for all components
        nodes = [i[0] for i in nx.connected_components(G)]
    else:
        # produce edges for components with source
        nodes = [source]
    # extra variables for cycle detection:
    cycle_stack = []
    output_cycles = set()

    def get_hashable_cycle(cycle):
        """cycle as a tuple in a deterministic order."""
        m = min(cycle)
        mi = cycle.index(m)
        mi_plus_1 = mi + 1 if mi < len(cycle) - 1 else 0
        if cycle[mi - 1] > cycle[mi_plus_1]:
            result = cycle[mi:] + cycle[:mi]
        else:
            result = list(reversed(cycle[:mi_plus_1])) + list(reversed(cycle[mi_plus_1:]))
        return tuple(result)

    for start in nodes:
        if start in cycle_stack:
            continue
        cycle_stack.append(start)

        stack = [(start, iter(G[start]))]
        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)

                if child not in cycle_stack:
                    cycle_stack.append(child)
                    stack.append((child, iter(G[child])))
                else:
                    i = cycle_stack.index(child)
                    if i < len(cycle_stack) - 2:
                        output_cycles.add(get_hashable_cycle(cycle_stack[i:]))

            except StopIteration:
                stack.pop()
                cycle_stack.pop()

    return [list(i) for i in output_cycles]


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
    # colors = nx.get_edge_attributes(triangle, 'color').values()
    # nx.draw_networkx(triangle, pos, with_labels=True, connectionstyle='arc3, rad = 0.1', edge_color=colors)
    cmap = plt.cm.viridis(np.linspace(0, 1, graph.number_of_edges()))
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos, with_labels=True, connectionstyle='arc3, rad = 0.1')



    # nx.draw_networkx_edges(triangle, pos, edge_color=cmap)
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

    triangle = nx.DiGraph()
    triangle.add_nodes_from(['Alice', 'Bob', 'Carol', 'Emma', 'Dave'])

    # Alice --> Bob and Alice <-- Bob
    triangle.add_edge('Alice', 'Bob', color='r')
    triangle.add_edge('Bob', 'Alice', color='g')
    # both sides
    triangle['Alice']['Bob']['initial_balance'] = 20
    triangle['Bob']['Alice']['initial_balance'] = 10
    # capacities
    triangle['Alice']['Bob']['satoshis'] = 30
    triangle['Bob']['Alice']['satoshis'] = 30
    # flows
    triangle['Alice']['Bob']['flow_bound'] = 5
    triangle['Bob']['Alice']['flow_bound'] = 0
    # objective
    triangle['Alice']['Bob']['objective function coefficient'] = 1

    # Bob --> Carol and Bob <-- Carol
    triangle.add_edge('Bob', 'Carol', color='g')
    triangle.add_edge('Carol', 'Bob', color='b')
    # both sides
    triangle['Bob']['Carol']['initial_balance'] = 40
    triangle['Carol']['Bob']['initial_balance'] = 10
    # capacities
    triangle['Bob']['Carol']['satoshis'] = 50
    triangle['Carol']['Bob']['satoshis'] = 50
    # flows
    triangle['Bob']['Carol']['flow_bound'] = 15
    triangle['Carol']['Bob']['flow_bound'] = 0
    # objective
    triangle['Alice']['Bob']['objective function coefficient'] = 1

    # Carol --> Alice and Carol <-- Alice
    triangle.add_edge('Carol', 'Alice', color='b')
    triangle.add_edge('Alice', 'Carol', color='r')
    # both sides
    triangle['Carol']['Alice']['initial_balance'] = 40
    triangle['Alice']['Carol']['initial_balance'] = 20
    # capacities
    triangle['Carol']['Alice']['satoshis'] = 60
    triangle['Alice']['Carol']['satoshis'] = 60
    # objective
    triangle['Carol']['Alice']['objective function coefficient'] = 1
    # flows
    triangle['Carol']['Alice']['flow_bound'] = 10
    triangle['Alice']['Carol']['flow_bound'] = 0

    # Alice --> Dave and Alice <-- Dave
    triangle.add_edge('Alice', 'Dave', color='r')
    triangle.add_edge('Dave', 'Alice', color='y')
    # both sides
    triangle['Alice']['Dave']['initial_balance'] = 70
    triangle['Dave']['Alice']['initial_balance'] = 30
    # capacities
    triangle['Alice']['Dave']['satoshis'] = 100
    triangle['Dave']['Alice']['satoshis'] = 100
    # flows
    triangle['Alice']['Dave']['flow_bound'] = 20
    triangle['Dave']['Alice']['flow_bound'] = 0
    # objective
    triangle['Alice']['Dave']['objective function coefficient'] = 1

    # Alice --> Emma and Alice <-- Emma
    triangle.add_edge('Alice', 'Emma', color='r')
    triangle.add_edge('Emma', 'Alice', color='m')
    # both sides
    triangle['Alice']['Emma']['initial_balance'] = 20
    triangle['Emma']['Alice']['initial_balance'] = 20
    # capacities
    triangle['Alice']['Emma']['satoshis'] = 40
    triangle['Emma']['Alice']['satoshis'] = 40
    # flows
    triangle['Alice']['Emma']['flow_bound'] = 0
    triangle['Emma']['Alice']['flow_bound'] = 0

    # Bob --> Dave and Bob <-- Dave
    triangle.add_edge('Bob', 'Dave', color='g')
    triangle.add_edge('Dave', 'Bob', color='y')
    # both sides
    triangle['Bob']['Dave']['initial_balance'] = 20
    triangle['Dave']['Bob']['initial_balance'] = 70
    # capacities
    triangle['Bob']['Dave']['satoshis'] = 90
    triangle['Dave']['Bob']['satoshis'] = 90
    # flows
    triangle['Bob']['Dave']['flow_bound'] = 0
    triangle['Dave']['Bob']['flow_bound'] = 25
    # objective
    triangle['Dave']['Bob']['objective function coefficient'] = 1

    # Bob --> Emma and Bob <-- Emma
    triangle.add_edge('Bob', 'Emma', color='g')
    triangle.add_edge('Emma', 'Bob', color='m')
    # both sides
    triangle['Bob']['Emma']['initial_balance'] = 50
    triangle['Emma']['Bob']['initial_balance'] = 60
    # capacities
    triangle['Bob']['Emma']['satoshis'] = 110
    triangle['Emma']['Bob']['satoshis'] = 110
    # flows
    triangle['Bob']['Emma']['flow_bound'] = 0
    triangle['Emma']['Bob']['flow_bound'] = 5
    # objective
    triangle['Emma']['Bob']['objective function coefficient'] = 1

    # Carol --> Dave and Carol <-- Dave
    triangle.add_edge('Carol', 'Dave', color='b')
    triangle.add_edge('Dave', 'Carol', color='y')
    # both sides
    triangle['Carol']['Dave']['initial_balance'] = 80
    triangle['Dave']['Carol']['initial_balance'] = 20
    # capacities
    triangle['Carol']['Dave']['satoshis'] = 100
    triangle['Dave']['Carol']['satoshis'] = 100
    # objective
    triangle['Carol']['Dave']['objective function coefficient'] = 1
    # flows
    triangle['Carol']['Dave']['flow_bound'] = 30
    triangle['Dave']['Carol']['flow_bound'] = 0

    # Carol --> Emma and Carol <-- Emma
    triangle.add_edge('Carol', 'Emma', color='b')
    triangle.add_edge('Emma', 'Carol', color='m')
    # both sides
    triangle['Carol']['Emma']['initial_balance'] = 10
    triangle['Emma']['Carol']['initial_balance'] = 90
    # capacities
    triangle['Carol']['Emma']['satoshis'] = 100
    triangle['Emma']['Carol']['satoshis'] = 100
    # objective
    triangle['Emma']['Carol']['objective function coefficient'] = 1
    # flows
    triangle['Carol']['Emma']['flow_bound'] = 0
    triangle['Emma']['Carol']['flow_bound'] = 40

    # Plotting
    # plot_graph_with_capacities(triangle)
    # plot_graph_with_initial_balances(triangle)

    balance_updates = LP_global_rebalancing(triangle)
    cycle_decomposition(balance_updates, triangle)



if __name__ == "__main__":
    main()
