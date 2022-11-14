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
                append_to_A(1, 2*m + i, edge_index, data, row, col)

                # ineq 2b: - \sum_{out edges} f(u,v)
                append_to_A(-1, 2*m + n + i, edge_index, data, row, col)

            for edge in rebalancing_graph.in_edges(u):
                edge_index = list_of_edges.index(edge)

                # ineq 2a: - \sum_{in edges} f(v,u)
                append_to_A(-1, 2*m + i, edge_index, data, row, col)

                # ineq 2b: \sum_{in edges} f(v,u)
                append_to_A(1, 2*m + n + i, edge_index, data, row, col)


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
def cycle_decomposition(balance_updates, rebalancing_graph):

    cycle_flows = []
    active_edges = []

    # consider only those edges that aren't null
    for i in range(len(balance_updates)):
        if balance_updates[i][2] != 0:
            active_edges.append(balance_updates[i])

    circulation_graph = nx.DiGraph()
    circulation_graph_weighted = nx.DiGraph()
    reduced_list_of_edges = list(map(lambda edge: (edge[0], edge[1]), active_edges))
    circulation_graph.add_edges_from(reduced_list_of_edges)
    circulation_graph_weighted.add_weighted_edges_from(active_edges)

    while len(active_edges) > 0:
        cycle = nx.find_cycle(circulation_graph)

        # w ←− min f(e),e ∈ Ci
        min_flow = min(dict(circulation_graph_weighted.edges).items(), key=lambda x: x[1]['weight'])


        for edge in cycle:
            # total_flow = sum(circulation_graph_weighted[u][v]['weight'] for (u, v) in test)
            smallest_flow = min_flow[edge]['weight']
            circulation_graph_weighted.remove_edge(smallest_flow)

            # f(e) ←− f(e) − f_i(e)
            # if f(e) = 0:
                # active_edges.remove(edge)

    return cycle_flows


def main():
    # The simplest graph
    #      (20) Alice (20)
    #           ^   ^
    #     (40) /      \  (10)
    #   Carol <--------> Bob
    #     (10)          (40)

    triangle = nx.DiGraph()
    triangle.add_nodes_from(['Alice', 'Bob', 'Carol'])

    # Alice --> Bob and Alice <-- Bob
    triangle.add_edge('Alice', 'Bob')
    triangle.add_edge('Bob', 'Alice')
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
    triangle.add_edge('Bob', 'Carol')
    triangle.add_edge('Carol', 'Bob')
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
    triangle.add_edge('Carol', 'Alice')
    triangle.add_edge('Alice', 'Carol')
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
    triangle.add_edge('Alice', 'Dave')
    triangle.add_edge('Dave', 'Alice')
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
    triangle.add_edge('Alice', 'Emma')
    triangle.add_edge('Emma', 'Alice')
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
    triangle.add_edge('Bob', 'Dave')
    triangle.add_edge('Dave', 'Bob')
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
    triangle.add_edge('Bob', 'Emma')
    triangle.add_edge('Emma', 'Bob')
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
    triangle.add_edge('Carol', 'Dave')
    triangle.add_edge('Dave', 'Carol')
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
    triangle.add_edge('Carol', 'Emma')
    triangle.add_edge('Emma', 'Carol')
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

    balance_updates = LP_global_rebalancing(triangle)
    cycle_decomposition(balance_updates, triangle)

if __name__ == "__main__":
    main()