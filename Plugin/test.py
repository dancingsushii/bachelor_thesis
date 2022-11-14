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
def cycle_decomposition(rebalancing_graph, balance_updates):
    # Balance updates will be received as a list
    # [('Charlie','Bob', 10), ('Bob','Alice', 4), ...]
    cycle_flows = []

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
    triangle['Carol']['Bob']['flow_bound'] = 5



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



    # HS_rebalancing_graph(HS)
    balances_updates = LP_global_rebalancing(triangle)
    cycle_decomposition(balances_updates, triangle)

if __name__ == "__main__":
    main()