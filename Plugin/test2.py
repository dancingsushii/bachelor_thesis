import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import networkx as nx
import concurrent.futures
import matplotlib.pyplot as plt

global rebalancing_graph
global list_of_nodes
global list_of_edges
list_of_nodes = list(rebalancing_graph.nodes)
list_of_edges = list(rebalancing_graph.edges)


# Step 5: Executes LP on that graph (need a license be deployed)
def LP_global_rebalancing(rebalancing_graph, num_of_conc_cores):


    try:
        n = rebalancing_graph.number_of_nodes()
        m = rebalancing_graph.number_of_edges()


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


        data = []
        row = []
        col = []
        rhs = np.zeros(2 * m + 2 * n)

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

        print(f'done with constraint 1')

        inputs_2 = [(n, m, node_index) for node_index in range(n)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_of_conc_cores) as executor:
            results_2 = executor.map(compute_costraint_2, inputs_2)  # results is a generator object

        print(f'done with constraint 2')

        all_entries_2 = [(d, r, c) for list_of_tuples in results_2 for d, r, c in list_of_tuples]
        all_entries += all_entries_2  # unpack generator

        extract_coord = lambda coord: [tpl[coord] for tpl in all_entries]

        data = extract_coord(0)
        row = extract_coord(1)
        col = extract_coord(2)

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


def compute_costraint_1a(edge_index):
    # -f(u,v) <= 0
    return (-1, edge_index, edge_index)


def compute_costraint_1b(args):
    # f(u,v) <= m(u,v)
    m, edge_index = args
    return (1, m + edge_index, edge_index)


def compute_costraint_2(args):


    n, m, node_index = args

    u = list_of_nodes[node_index]

    triples = []

    for edge in rebalancing_graph.out_edges(u):
        edge_index = list_of_edges.index(edge)

        # ineq 2a: \sum_{out edges} f(u,v)
        triples.append((1, 2 * m + node_index, edge_index))

        # ineq 2b: - \sum_{out edges} f(u,v)
        triples.append((-1, 2 * m + n + node_index, edge_index))

    for edge in rebalancing_graph.in_edges(u):
        edge_index = list_of_edges.index(edge)

        # # ineq 2a: - \sum_{in edges} f(v,u)
        triples.append((-1, 2 * m + node_index, edge_index))

        # # ineq 2b: \sum_{in edges} f(v,u)
        triples.append((1, 2 * m + n + node_index, edge_index))

    return triples

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


    num_of_conc_cores = 3
    LP_global_rebalancing(HS, num_of_conc_cores)

if __name__ == "__main__":
    main()

