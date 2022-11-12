import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt

# Step 5: Executes LP on that graph (need a license be deployed)
def HS_rebalancing_graph(topology, channels_where_trx_failed):

    rebalancing_graph = topology
    max_flow = max(
        {topology[x][y]['initial balance'] - topology[x][y]['satoshis'] for x, y, z in channels_where_trx_failed})
    print(f'm(u,v) variables set to {max_flow}')

    # Consider only those edges whose nodes are in "participating" state
    # setup channels and maximum flow_bound
    rebalancing_graph_edges = [(u, v) for u, v in topology.edges]
    rebalancing_graph.add_edges_from(rebalancing_graph_edges, flow_bound=max_flow)

    # if a transaction failed at (src,dst), set 0 flow to this direction and let max_flow in the reverse
    for src, dst, val in channels_where_trx_failed:
        rebalancing_graph[src][dst]['flow_bound'] = 0
        rebalancing_graph[dst][src]['objective function coefficient'] = 1

    # set objective function coefficients
    # we are never here bcs state== 'depleted' is commented out
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
    # for channel in channels_where_trx_failed:
    #     u,v = channel
    #     rebalancing_graph.add_edge(u,v)

    #     # optimally, the amount of flow through (u,v) should be the satoshis needed such that (v,u),
    #     # the channel to be refunded, returns to its initial balance
    #     rebalancing_graph[u][v]['max flow'] = topology[v][u]['initial balance'] - topology[v][u]['satoshis']

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

        # only channels where transactions failed contribute to the objective function
        # for edge_index in range(m):
            # u, v = list_of_edges[edge_index]
            # if 'objective function coefficient' in rebalancing_graph[u][v]:
                # obj[edge_index] = rebalancing_graph[u][v]

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
            u,v = list_of_edges[edge_index]

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


# Step 6: Cycle decomposition


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


def main():
    G = nx.DiGraph()

    G.add_nodes_from(['A', 'B', 'C', 'D', 'F'])
    G.add_edge('A', 'B', flow_bound='30')
    G.add_edge('B', 'A', flow_bound='0')
    G['B']['A']['initial_balance'] = 80
    G['A']['A']['initial_balance'] = 80
    G['B']['A']['objective function coefficient'] = 1

    G.add_edge('A', 'C', flow_bound='0')
    G.add_edge('C', 'A', flow_bound='32.5')
    G['C']['A']['initial_balance'] = 65
    G['A']['C']['initial_balance'] = 65
    G['A']['C']['objective function coefficient'] = 1

    G.add_edge('A', 'D', flow_bound='0')
    G.add_edge('D', 'A', flow_bound='30')
    G['D']['A']['initial_balance'] = 80
    G['A']['D']['initial_balance'] = 80
    G['A']['D']['objective function coefficient'] = 1

    G.add_edge('B', 'C', flow_bound='0')
    G.add_edge('C', 'B', flow_bound='20')
    G['B']['C']['initial_balance'] = 80
    G['C']['B']['initial_balance'] = 80
    G['B']['C']['objective function coefficient'] = 1

    G.add_edge('C', 'D', flow_bound='20')
    G.add_edge('D', 'C', flow_bound='0')
    G['D']['C']['initial_balance'] = 100
    G['C']['D']['initial_balance'] = 100
    G['D']['C']['objective function coefficient'] = 1

    G.add_edge('F', 'C', flow_bound='42.5')
    G.add_edge('C', 'F', flow_bound='0')
    G['F']['C']['initial_balance'] = 85
    G['C']['F']['initial_balance'] = 85
    G['C']['F']['objective function coefficient'] = 1

    G.add_edge('F', 'B', flow_bound='0')
    G.add_edge('B', 'F', flow_bound='10')
    G['B']['F']['objective function coefficient'] = 1
    G['F']['B']['initial_balance'] = 50
    G['B']['F']['initial_balance'] = 50
    channels_where_trx_failed = [('B', 'C', 30), ('B', 'F', 40), ('C', 'D', 75)]
    HS_rebalancing_graph(G, channels_where_trx_failed)
    LP_global_rebalancing(G)

if __name__ == "__main__":
    main()