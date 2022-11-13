import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt

# Step 5: Executes LP on that graph (need a license be deployed)
def HS_rebalancing_graph(topology):

    rebalancing_graph = topology

    # m(u, v) - 'flow_bound'
    # max_flow = max(
    #     {topology[x][y]['initial_balance'] - topology[x][y]['satoshis'] for x, y in topology})
    # print(f'm(u,v) variables set to {max_flow}')

    max_flow = max({topology[x][y]['flow_bound'] for x, y in topology})

    # Consider only those edges whose nodes are in "participating" state
    # setup channels and maximum flow_bound
    rebalancing_graph_edges = [(u, v) for u, v in topology.edges]
    rebalancing_graph.add_edges_from(rebalancing_graph_edges, flow_bound=max_flow)

    # if a transaction failed at (src,dst), set 0 flow to this direction and let max_flow in the reverse
    for src, dst, val in topology:
        rebalancing_graph[src][dst]['flow_bound'] = 0
        rebalancing_graph[dst][src]['objective function coefficient'] = 1

    # set objective function coefficients, we are never here bcs state== 'depleted' is commented out
    for u, v in topology.edges:

        # adjust flow bounds if they exceed a preset limit
        if rebalancing_graph[u][v]['flow_bound'] > topology[u][v]['satoshis'] / 2:
            rebalancing_graph[u][v]['flow_bound'] = topology[u][v]['satoshis'] / 2

    return rebalancing_graph


def LP_global_rebalancing(rebalancing_graph):

    try:
        n = rebalancing_graph.number_of_nodes()
        m = rebalancing_graph.number_of_edges()
        global list_of_nodes
        global list_of_edges
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


# Step 6: Cycle decomposition


def main():
    G = nx.DiGraph()

    G.add_nodes_from(['A', 'B', 'C', 'D', 'F'])
    G.add_edge('A', 'B')
    G.add_edge('B', 'A')
    G['B']['A']['initial_balance'] = 10
    G['A']['B']['initial_balance'] = 70
    G['B']['A']['satoshis'] = 10
    G['A']['B']['satoshis'] = 70
    # G['B']['A']['objective function coefficient'] = 1

    G.add_edge('A', 'C')
    G.add_edge('C', 'A')
    G['C']['A']['initial_balance'] = 60
    G['A']['C']['initial_balance'] = 5
    G['C']['A']['satoshis'] = 60
    G['A']['C']['satoshis'] = 5
    # G['A']['C']['objective function coefficient'] = 1

    G.add_edge('A', 'D')
    G.add_edge('D', 'A')
    G['D']['A']['initial_balance'] = 70
    G['A']['D']['initial_balance'] = 10
    G['D']['A']['satoshis'] = 70
    G['A']['D']['satoshis'] = 10
    # G['A']['D']['objective function coefficient'] = 1

    G.add_edge('B', 'C')
    G.add_edge('C', 'B')
    G['B']['C']['initial_balance'] = 20
    G['C']['B']['initial_balance'] = 60
    G['B']['C']['satoshis'] = 20
    G['C']['B']['satoshis'] = 60
    # G['B']['C']['objective function coefficient'] = 1

    G.add_edge('C', 'D')
    G.add_edge('D', 'C')
    G['D']['C']['initial_balance'] = 30
    G['C']['D']['initial_balance'] = 70
    G['D']['C']['satoshis'] = 30
    G['C']['D']['satoshis'] = 70
    # G['D']['C']['objective function coefficient'] = 1

    G.add_edge('F', 'C')
    G.add_edge('C', 'F')
    G['F']['C']['initial_balance'] = 80
    G['C']['F']['initial_balance'] = 5
    G['F']['C']['satoshis'] = 80
    G['C']['F']['satoshis'] = 5
    # G['C']['F']['objective function coefficient'] = 1

    G.add_edge('F', 'B')
    G.add_edge('B', 'F')
    # G['B']['F']['objective function coefficient'] = 1
    G['F']['B']['initial_balance'] = 20
    G['B']['F']['initial_balance'] = 30
    G['F']['B']['satoshis'] = 20
    G['B']['F']['satoshis'] = 30
    # channels_where_trx_failed = [('B', 'C', 30), ('B', 'F', 40), ('C', 'D', 75)]

    # Graph from H&S paper
    HS = nx.DiGraph()
    HS.add_nodes_from(['Charlie', 'Bob', 'Alice', 'Dave'])

    # Charlie --> Bob and Charlie <-- Bob
    HS.add_edge('Charlie', 'Bob')
    HS.add_edge('Bob', 'Charlie')
    # initial balances on both sides
    HS['Charlie']['Bob']['initial_balance'] = 30
    HS['Bob']['Charlie']['initial_balance'] = 10
    # capacities
    HS['Bob']['Charlie']['satoshis'] = 40
    HS['Charlie']['Bob']['satoshis'] = 40
    # manual flow_bounds
    HS['Charlie']['Bob']['flow_bound'] = 10
    HS['Bob']['Charlie']['flow_bound'] = 0
    HS['Charlie']['Bob']['objective function coefficient'] = 1

    # Alice --> Bob and Alice <-- Bob
    HS.add_edge('Alice', 'Bob')
    HS.add_edge('Bob', 'Alice')
    HS['Alice']['Bob']['initial_balance'] = 10
    HS['Bob']['Alice']['initial_balance'] = 4
    HS['Bob']['Alice']['satoshis'] = 14
    HS['Alice']['Bob']['satoshis'] = 14
    HS['Alice']['Bob']['flow_bound'] = 3
    HS['Bob']['Alice']['flow_bound'] = 0
    HS['Alice']['Bob']['objective function coefficient'] = 1


    # Bob --> Dave and Dave <-- Bob
    HS.add_edge('Bob', 'Dave')
    HS.add_edge('Dave', 'Bob')
    HS['Dave']['Bob']['initial_balance'] = 10
    HS['Bob']['Dave']['initial_balance'] = 6
    HS['Bob']['Dave']['satoshis'] = 16
    HS['Dave']['Bob']['satoshis'] = 16
    HS['Dave']['Bob']['flow_bound'] = 2
    HS['Bob']['Dave']['flow_bound'] = 0
    HS['Dave']['Bob']['objective function coefficient'] = 1

    # Alice --> Dave and Dave <-- Alice
    HS.add_edge('Alice', 'Dave')
    HS.add_edge('Dave', 'Alice')
    HS['Alice']['Dave']['initial_balance'] = 10
    HS['Dave']['Alice']['initial_balance'] = 6
    HS['Dave']['Alice']['satoshis'] = 16
    HS['Alice']['Dave']['satoshis'] = 16
    HS['Alice']['Dave']['flow_bound'] = 2
    HS['Dave']['Alice']['flow_bound'] = 0
    HS['Alice']['Dave']['objective function coefficient'] = 1

    # Alice --> Charlie and Alice <-- Charlie
    HS.add_edge('Alice', 'Charlie')
    HS.add_edge('Charlie', 'Alice')
    HS['Alice']['Charlie']['initial_balance'] = 10
    HS['Charlie']['Alice']['initial_balance'] = 6
    HS['Charlie']['Alice']['satoshis'] = 16
    HS['Alice']['Charlie']['satoshis'] = 16
    HS['Charlie']['Alice']['flow_bound'] = 2
    HS['Alice']['Charlie']['flow_bound'] = 0
    HS['Charlie']['Alice']['objective function coefficient'] = 1

    # channels_where_trx_failed = [('Bob', 'Dave', 10), ('Dave', 'Alice', 10), ('Bob', 'Alice', 10)]

    # HS_rebalancing_graph(HS)
    LP_global_rebalancing(HS)

if __name__ == "__main__":
    main()