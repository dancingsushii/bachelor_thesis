#!/usr/bin/env python3

from pyln.client import Plugin, RpcError, Millisatoshi
import json, threading, random, os
import networkx as nx
from queue import Queue, Empty
from enum import Enum
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import ast
import time
import uuid

# TODO increment msgid by one for replies that signal the next state
# TODO write one method to send messages
# TODO clean up f"" and "".format()
# TODO add all attributes from params region as plugin options 


# region params
MESSAGE_BUS_TOPIC = 'hide_and_seek_message_bus'
REBALANCE_REQ_TIMEOUT_S = 60
REBALANCE_RES_TIMEOUT_S = 60 * 10
REBALANCE_REQ_SEARCH_DEPTH = 5
# endregion


plugin = Plugin()
busyRebalancing = threading.Event()


# region networking and request dispatching
class HideSeekMsgCode(Enum):
    """Status codes for messages sent in hide & seek rebalancing process.
    Numbers should be ``hex`` and ``odd``"""
    REBALANCE_REQ_MSG_HEX = 0xFFFF
    # notify peer req to him timed out #
    REBALANCE_REQ_INTERRUPT_MSG_HEX = 0x4165
    # response to request of initial phase of hide and seek #
    REBALANCE_RES_MSG_HEX = 0x3FCF
    # short participation replies #
    REBALANCE_ACC_MSG_HEX = 0x4161
    REBALANCE_REJ_MSG_HEX = 0x4163

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def get_key(cls, value):
        if cls.has_value:
            return cls._value2member_map_[value]


class RebalanceMessage:

    def __init__(self, peer: str, msg_id: str, msg_type: int, payload):
        self.peer = peer
        self.msg_id = msg_id
        self.msg_type = HideSeekMsgCode.get_key(msg_type)
        self.payload = payload


class RebalanceRequestBody:

    def __init__(self, initiator_id: str, participants: list, rejectors: list, search_depth: int,
                 accum_depth: int) -> None:
        self.initiator_id = initiator_id
        self.participants = participants
        self.rejectors = rejectors
        self.search_depth = search_depth
        self.accum_depth = accum_depth

    def toString(self):
        payload_body = {
            "initiator_id": self.initiator_id,
            "participants": self.participants,
            "rejectors": self.rejectors,
            "search_depth": self.search_depth,
            "accum_depth": self.accum_depth
        }
        return json.dumps(payload_body)

    @classmethod
    def fromString(cls, str):
        payload_body = json.loads(str)
        return RebalanceRequestBody(
            payload_body["initiator_id"],
            payload_body["participants"],
            payload_body["rejectors"],
            payload_body["search_depth"],
            payload_body["accum_depth"])


@plugin.async_hook('custommsg')
def on_custommsg(peer_id, payload, plugin, request, **kwargs):
    """Use custommsg hook to receive payload.
    Pass messages of type ``HideSeekMsgCode`` to ``MESSAGE_BUS_TOPIC`` channel for processing."""
    pbytes = bytes.fromhex(payload)
    mtype = int.from_bytes(pbytes[:2], "big")
    if HideSeekMsgCode.has_value(mtype):
        msgid = int.from_bytes(pbytes[2:10], "big")
        data = pbytes[10:].decode()
        message = dict({'peer': peer_id, 'msgid': hex(msgid), 'mtype': mtype, 'data': data})
        plugin.notify(MESSAGE_BUS_TOPIC, json.dumps(message))
    return request.set_result({'result': 'continue'})


@plugin.subscribe(MESSAGE_BUS_TOPIC)
def on_newmessage(plugin, origin, payload, request, **kwargs):
    """Dispatches incoming requests to request handlers and incoming responses into ResponseHandler queues."""
    d = json.loads(payload)
    message = RebalanceMessage(d['peer'], d['msgid'], d['mtype'], d['data'])
    plugin.log(
        f"Received message {message.msg_id} of type {message.msg_type.value} from {message.peer}: {message.payload}")
    if response_handler.is_response(message.msg_id):
        plugin.log(f"Received message is a response to existing request {message.msg_id}")
        response_handler.add_response(message.msg_id, message)
    elif message.msg_type == HideSeekMsgCode.REBALANCE_REQ_MSG_HEX:
        plugin.log('Received msg is a rebalance request.')
        handle_incoming_hide_seek_request(plugin, message)
    return


class ResponseHandler:

    def __init__(self) -> None:
        """Thread safe handling of message responses for hide&seek in lightning.
    
    Methods:\n
    ``open_request(request_id:str)`` -- Prepare a queue for responses to your request. Call it after sending a request you want to track response to.\n
    ``add_response(request_id:str, response)`` -- Add response to the response-queue of the request. Call it from the main thread (that handles incoming messages) to track response. Use msg-id as the first param.\n
    ``get_response(request_id:str, timeout:int)`` -- Pop the earliest response to your request. Blocks the thread and throws an exception if timeout occurs.\n
    ``close_request(request_id:str)`` -- Close request and process no responses to it anymore.\n
    ``is_valid(response)`` -- for a given response checks, if it is valid
    """
        self._responses = dict()
        self._remove_lock = threading.Lock()

    def open_request(self, request_id: str):
        """Prepare a queue for responses to your request."""
        if request_id not in self._responses:
            self._responses[request_id] = Queue()

    def add_response(self, request_id: str, response: RebalanceMessage):
        """Add response to the response-queue of the request.
    
    Keyword arguments:\n
    request_id -- id of the corresponding request \n
    response -- response message.
    """
        self._responses[request_id].put(response)

    def get_response(self, request_id: str, timeout) -> RebalanceMessage:
        """Dequeue a response to your request.
    
    Keyword arguments:\n
    request_id -- id of the corresponding request \n
    timeout -- how long to wait for a response

    Returns:\n
    response -- response object. Should contain fields: \n
    ``peer``, ``msg_id``, ``msg_type``, ``payload``

    Throws:\n
    Empty exception -- if no response comes in timeout seconds time
    """
        return self._responses[request_id].get(block=True, timeout=timeout)

    def is_response(self, response_id: str) -> bool:
        """Check if incoming message id is being awaited as a response to a happened request."""
        return response_id in self._responses

    def close_request(self, request_id: str):
        """Close request and process no responses to it anymore"""
        with self._remove_lock:
            if request_id not in self._responses:
                return
            request_responses = self._responses.pop(request_id)
            with request_responses.mutex:
                request_responses.queue.clear()


# endregion

# region handling and logic


def prepare_hide_seek(plugin, source_request: RebalanceMessage = None):
    """Hide & Seek data preparation handler.\n
    Differentiates between initiator and normal node by presence of the source_request:\n
    if source_request is present, the node received the request from another node. If not -> the node is the initiator."""
    plugin.log('Daemon initiating hide and seek...')
    own_node_id = get_node_id(plugin)
    rebalancing_graph = nx.DiGraph()
    if source_request is not None:
        # this node is not the initiator of rebalancing
        available_data = RebalanceRequestBody.fromString(source_request.payload)
        participants = available_data.participants + [own_node_id]
        rejectors = available_data.rejectors
        search_depth = available_data.search_depth
        accum_depth = available_data.accum_depth + 1
    else:
        # this node is the initiator
        participants = [own_node_id]
        rejectors = []
        search_depth = REBALANCE_REQ_SEARCH_DEPTH
        accum_depth = 0
    if accum_depth < search_depth:
        # send further requests and collect data from neighbors
        collect_neighbors_replies(plugin, participants, rejectors, search_depth, accum_depth, rebalancing_graph)
    else:
        plugin.log(f'Required search depth achieved, I can resend my network data available to me.')
        plugin.log(f"All neighbors processed or search deptch achieved. Extending rebalancing graph with local data...")
        # when no neighbors are left, add locally available information to the rebalancing_graph
        extend_rebalancing_graph_with_local_data(plugin, own_node_id, rebalancing_graph)
    if source_request is not None:
        # and send it back to the requestor
        plugin.log(f"Sending rebalancing graph back to the requestor...")
        send_rebalance_res(plugin, source_request, rebalancing_graph)
    else:
        # it is the delegate/initiator node, log the graph structure features for now
        final_statement = "COLLECTED GRAPH INFORMATION:\n"
        final_statement += "---------"
        for edge in rebalancing_graph.edges.data():
            src, dst = edge[0], edge[1]
            funding = edge[2]["funding_msat"]
            scid = edge[2]["scid"]
            edge_information = f"Edge {scid} from {src} towards {dst} with balance {funding}\n"
            final_statement += edge_information
            plugin.log(final_statement)


def collect_neighbors_replies(plugin, participants: list, rejectors: list, search_depth: int, accum_depth: int,
                              rebalancing_graph: nx.DiGraph):
    """Goes through neighbors, sends rebalance out requests and collects their local graphs into the common ``rebalancing_graph``"""
    neighbor_ids = get_neighbor_ids(plugin)
    for neighbor_id in neighbor_ids:
        if neighbor_id in participants or neighbor_id in rejectors:
            plugin.log(f"Neighbor {neighbor_id} already received invititation to participate.")
            continue
        plugin.log('Daemon picking {} to send rebalance out'.format(neighbor_id))
        try:
            request_body = RebalanceRequestBody(get_node_id(plugin), participants, rejectors, search_depth, accum_depth)
            # send rebalance_request to neighbor
            request_id = send_rebalance_out(plugin, neighbor_id, request_body)
            # wait until request times out or neighbor acknowledges
            response = response_handler.get_response(request_id, REBALANCE_REQ_TIMEOUT_S)
            # if peer rejected
            if response.msg_type == HideSeekMsgCode.REBALANCE_REJ_MSG_HEX:
                plugin.log('Peer {} responded with an REJ to request {}'.format(response.peer, request_id))
                rejectors.append(neighbor_id)
                continue
            if response.msg_type != HideSeekMsgCode.REBALANCE_ACC_MSG_HEX:
                plugin.log(
                    f'Received a response of unexpected type {response.msg_id} as a reply to request {request_id}')
                continue
            # ... peer accepted then
            plugin.log(
                'Peer {} responded with an ACK to request {}. Waiting for the graph data from the peer now...'.format(
                    response.peer, request_id))
            # wait for neighbors rebalance response with rebalancing data
            response = response_handler.get_response(request_id, REBALANCE_RES_TIMEOUT_S)
            # collect graph data from the response
            plugin.log(f"Graph data arrived! Packing into the rebalancing graph!")
            filename = "rebalancing_graph.gpickle.gz"
            if os.path.exists(filename): os.remove(filename)
            with open(filename, "wb") as f:
                f.write(bytes.fromhex(response.payload))
            plugin.log(
                f"Graph data from host {neighbor_id} written into the file {filename}. Reading graph data from there...")
            peers_rebalancing_graph: nx.DiGraph = nx.read_gpickle(filename)
            plugin.log(
                f"Delivered graph contains {len(peers_rebalancing_graph.nodes)} nodes and {len(peers_rebalancing_graph.edges)} edges")
            # merge the existing rebalancing_graph with TODO read this methods docks after accomplishing the correct data collection
            rebalancing_graph = nx.compose(rebalancing_graph, peers_rebalancing_graph)
            plugin.log(
                f"Composed graph contains {len(rebalancing_graph.nodes)} nodes and {len(rebalancing_graph.edges)} edges")
        # return collected graph data
        except Empty:
            # handle timeout
            plugin.log('Request {} timed out'.format(request_id))
        except Exception as e:
            # handle other errors
            plugin.log(
                'An unexpected error occured during request handling of request {}. Exception information: \n{}\n'.format(
                    request_id, str(e)))
        finally:
            # handle finish TODO
            # response_handler.close_request(request_id=request_id)
            plugin.log('Closed response handler for request {}'.format(request_id))


def handle_incoming_hide_seek_request(plugin, request: RebalanceMessage):
    # check if node is not participating in rebalancing already
    if busyRebalancing.is_set():
        plugin.log("Ignore request to start hide and seek. Apperently I am already participating in rebalancing.")
        return
    else:
        busyRebalancing.set()
    # decide using nodes own objection function if rebalancing makes sense
    if not decide_to_participate(plugin):
        plugin.log("There is no need to participate in rebalancing for me.")
        send_rebalance_rej(plugin, request.peer)
        return
    # simply send ACK for now, need to check if ResponseHandler works properly
    send_rebalance_acc(plugin, request.peer, request.msg_id)
    plugin.log('Request ACK sent to {}'.format(request.peer))
    # start rebalancing deamon
    plugin.log('Starting the rebalancing deamon for hide and seek...')
    threading.Thread(name="initiator", target=prepare_hide_seek, args=(plugin, request)).start()


def send_rebalance_out(plugin, peer_id, request_body: RebalanceRequestBody) -> int:
    """Sends rebalance out request to given peer_id. Messages are in the byte format of [type(2)][id(8)][data]."""
    payload = request_body.toString()
    msgtype = HideSeekMsgCode.REBALANCE_REQ_MSG_HEX.value
    msgid = random.randint(0, 2 ** 64)
    msg = (msgtype.to_bytes(2, 'big')
           + msgid.to_bytes(8, 'big')
           + bytes(payload, encoding='utf8'))
    request_id = hex(msgid)
    response_handler.open_request(request_id=request_id)
    res = plugin.rpc.call('sendcustommsg', {'node_id': peer_id, 'msg': msg.hex()})
    plugin.log('Request {} sent, rpc call result {}'.format(request_id, res))
    return request_id


def send_rebalance_rej(plugin, peer_id):
    plugin.log('Sending rebalance rej to peer_id {}'.format(peer_id))


def send_rebalance_acc(plugin, peer_id, request_id):
    """Sends rebalance ACK request to given peer_id. Messages are in the byte format of [type(2)][id(8)][data]."""
    payload = RebalanceRequestBody(peer_id, [], [], 1, 1).toString()
    msgtype = HideSeekMsgCode.REBALANCE_ACC_MSG_HEX.value
    msgid = int(request_id, base=0)
    msg = (msgtype.to_bytes(2, 'big')
           + msgid.to_bytes(8, 'big')
           + bytes(payload, encoding='utf8'))
    request_id = hex(msgid)
    res = plugin.rpc.call('sendcustommsg', {'node_id': peer_id, 'msg': msg.hex()})
    plugin.log('Request {} sent, rpc call result {}'.format(request_id, res))


def send_rebalance_res(plugin, source_request: RebalanceMessage, rebalancing_graph: nx.DiGraph):
    # pickle and compress the graph into the response message
    filename = f"rebalancing_graph.gpickle.gz"
    if os.path.exists(filename): os.remove(filename)
    nx.write_gpickle(rebalancing_graph, filename)
    plugin.log(
        f"Complete graph of {len(rebalancing_graph.nodes)} nodes and {len(rebalancing_graph.edges)} edges created and written into {filename}")
    # write it into the rebalancing response
    with open(filename, "rb") as f:
        payload = (f.read()).hex()
        msgtype = HideSeekMsgCode.REBALANCE_RES_MSG_HEX.value
        msgid = source_request.msg_id
        msg = (msgtype.to_bytes(2, 'big')
               + msgid.to_bytes(8, 'big')
               + bytes(payload, encoding='utf8'))
        request_id = hex(msgid)
        res = plugin.rpc.call('sendcustommsg', {'node_id': source_request.peer, 'msg': msg.hex()})
        plugin.log(f'Request {request_id} sent, rpc call result {res}')
    return request_id


# endregion

# region graph logic


def get_node_id(plugin):
    return plugin.rpc.getinfo()["id"]


def get_neighbor_ids(plugin):
    node_id = get_node_id(plugin)
    # get information on your connected peers
    neighbors = plugin.rpc.call('listchannels', {'source': node_id})
    # retrieve their ids
    neighbor_ids = map(lambda channel: channel["destination"], neighbors["channels"])
    return neighbor_ids


def get_peers(plugin):
    """Returns infos about peers in a list of dicts. Only peers with opened channels in ``CHANNELD_NORMAL`` state and with "connected" == True are returned. \nFind an example of one dict in the returned list in the peer_example.json."""
    # get channels dict
    peers = plugin.rpc.call('listpeers', {})
    # filter channels
    filter_strategy = lambda peer: "channels" in peer and len(peer["channels"]) > 0 \
                                   and peer["channels"][0]["state"] == "CHANNELD_NORMAL" and peer["connected"] == True
    # return filtered results
    return list(filter(filter_strategy, peers["peers"]))


def decide_to_participate(plugin) -> bool:
    return True


def extend_rebalancing_graph_with_local_data(plugin, own_id, rebalancing_graph: nx.DiGraph):
    """Extends the rebalancing graph with the required local graph information"""
    peers = get_peers(plugin)
    peer_ids = list(map(lambda peer: peer["id"], peers))
    plugin.log(f"Collecting local rebalancing graph... collected peer_ids {peer_ids}")
    for peer in peers:
        peer_id, channel = peer["id"], peer["channels"][0]
        own_funding, remote_funding, scid = channel["funding_msat"][own_id], channel["funding_msat"][peer_id], channel[
            "short_channel_id"]
        check_and_add_edge(rebalancing_graph, src=own_id, dst=peer_id, funding_msat=own_funding, scid=scid)
        check_and_add_edge(rebalancing_graph, src=peer_id, dst=own_id, funding_msat=remote_funding, scid=scid)


def check_and_add_edge(graph: nx.DiGraph, src: str, dst: str, funding_msat: str, scid: str):
    # check if rebalancing_graph doesn't contain this channel already
    if graph.has_edge(src, dst): return
    # add channel as edge to our rebalancing_graph
    funding_msat_value = int(funding_msat.replace("msat", ""))
    graph.add_edge(src, dst, funding_msat=funding_msat_value, scid=scid)


# endregion

# region MPC blackbox


def LP_global_rebalancing(rebalancing_graph):
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


def cycle_decomposition(balance_updates, rebalancing_graph):
    # Step 6: Cycle decomposition on MPC delegate
    # Ask maybe we should clean balance_updates before?
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


# endregion


# region HTLC creation for cycles
def scid_from_peer(plugin, peer_id):
    """Returns ``scid`` for the channel with a given peer"""
    # get channels dict
    peers = plugin.rpc.listpeers(peer_id).get('peers')
    plugin.log(f"LOG in the function scid_from_peer {peers}")
    # filter channels
    filter_strategy = lambda peer: peer["id"] == peer_id and "channels" in peer and len(peer["channels"]) > 0 \
                                   and peer["channels"][0]["state"] == "CHANNELD_NORMAL" and peer["connected"] == True
    # return filtered results
    plugin.log(f"LOG in about filetered peers in the function scid_from_peer {filter_strategy}")
    return next(filter(filter_strategy, peers["peers"]), None)["channels"][0]["short_channel_id"]


def peer_from_scid(plugin, short_channel_id, my_node_id, payload):
    channels = plugin.rpc.listchannels(short_channel_id).get('channels')
    for ch in channels:
        if ch['source'] == my_node_id:
            return ch['destination']
    raise RpcError("rebalance", payload, {'message': 'Cannot find peer for channel: ' + short_channel_id})


def get_channel(plugin, payload, peer_id, scid, check_state: bool = False):
    peer = plugin.rpc.listpeers(peer_id).get('peers')[0]
    channel = next(c for c in peer['channels'] if c.get('short_channel_id') == scid)
    if check_state:
        if channel['state'] != "CHANNELD_NORMAL":
            raise RpcError('rebalance', payload,
                           {'message': 'Channel %s not in state CHANNELD_NORMAL, but: %s' % (scid, channel['state'])})
        if not peer['connected']:
            raise RpcError('rebalance', payload, {'message': 'Channel %s peer is not connected.' % scid})
    return channel


def cleanup(plugin, label, payload, rpc_result, error=None):
    try:
        # RFC Delete invoice
        plugin.rpc.delinvoice(label, 'unpaid')
    except RpcError as e:
        # race condition: waitsendpay timed out, but invoice get paid
        if 'status is paid' in e.error.get('message', ""):
            return rpc_result

    if error is not None and isinstance(error, RpcError):
        # unwrap rebalance errors as 'normal' RPC result
        if error.method == "rebalance":
            return {"status": "exception",
                    "message": error.error.get('message', "error not given")}
        raise error

    return rpc_result


@plugin.method('htlc_creation_for_cycle')
def htlc_creation_for_cycle(plugin, cycle_string: str, retry_for: int = 60):
    """Method for cyclic creation of HTLC.

    Is initiated randomly on one of the cycle participant after 1LP calculation.
    """
    plugin.log('Parsing string to list...')
    cycle = list(ast.literal_eval(cycle_string))

    plugin.log('Starting htlc_creation_for_cycles method...')
    # We are on Alice...

    # For this cycle we will input the necessary data but in real setup we will get it from another node
    # cycle = [(alice_id, carol_id, 25000),(carol_id, bob_id, 25000), (bob_id, alice_id, 25000)]
    # cycle[0] = (alice_id, carol_id, 25000)
    # cycle[1] = (carol_id, bob_id, 25000)
    # cycle[2] = (bob_id, alice_id, 25000)

    msatoshi = cycle[0][2]
    if msatoshi:
        msatoshi = Millisatoshi(msatoshi)

    my_node_info = plugin.rpc.getinfo()
    plugin.log(f"My node info is {my_node_info}")

    plugin.log("TEST LISTPEERS")
    list_peers = plugin.rpc.listpeers()['peers']
    plugin.log(f"KOSTJA {list_peers}")

    plugin.log(f"MY PRINT IS {cycle[0][1]}")
    outgoing_scid = scid_from_peer(plugin, cycle[0][1])
    plugin.log(f"Outgoing scid to {cycle[0][1]} is {outgoing_scid}")

    plugin.log(f"MY PRINT 2 IS {cycle[-1][0]}")
    incoming_scid = scid_from_peer(plugin, cycle[-1][0])
    plugin.log(f"Ingoing scid from {cycle[-1][0]} is {incoming_scid}")

    payload = {
        "outgoing_scid": outgoing_scid,
        "incoming_scid": incoming_scid,
        "msatoshi": msatoshi,
        "retry_for": retry_for,
    }

    my_node_id = plugin.rpc.getinfo().get('id')
    outgoing_node_id = peer_from_scid(plugin, outgoing_scid, my_node_id, payload)
    incoming_node_id = peer_from_scid(plugin, incoming_scid, my_node_id, payload)

    get_channel(plugin, payload, outgoing_node_id, outgoing_scid, True)
    get_channel(plugin, payload, incoming_node_id, incoming_scid, True)

    route_out = {'id': outgoing_node_id, 'channel': outgoing_scid}
    route_in = {'id': my_node_id, 'channel': incoming_scid}

    plugin.log('Creating timelock for cycle...')
    start_ts = int(time.time())
    label = "Rebalance-" + str(uuid.uuid4())
    description = "%s to %s" % (outgoing_scid, incoming_scid)

    plugin.log('Creating invoice for the whole cycle...')
    invoice = plugin.rpc.invoice(msatoshi, label, description)

    plugin.log('Creating payment_secret for the whole cycle...')
    payment_secret = invoice.get('payment_secret')

    plugin.log('Сreating a hash for this secret for the whole cycle...')
    payment_hash = invoice['payment_hash']

    try:
        while int(time.time()) - start_ts < retry_for:

            plugin.log(f"Setting the timer for {time.time() - start_ts} being less then {retry_for}")

            plugin.log('Trying to construct the midroute from cycle...')

            # edges_mid = (carol_id, bob_id, 25000)
            edges_mid = cycle[1:-1]

            if edges_mid[0]:
                edges_mid[0] = Millisatoshi(msatoshi)

            plugin.log('Parse the edges in edges_mid to route...')
            rpc_result = {
                "sent": msatoshi,
                "received": msatoshi,
                "outgoing_scid": outgoing_scid,
                "incoming_scid": incoming_scid,
                "status": "complete",
                "message": f"{msatoshi} sent over {len(route)} hops to rebalance {msatoshi}",
            }

            route_mid = r['route']
            route = [route_out] + route_mid + [route_in]

            for r in route:

                time_start = time.time()
                plugin.log("    - %s  %14s  %s" % (r['id'], r['channel'], r['amount_msat']), 'debug')

                try:
                    plugin.log('Executing sendpay...')
                    plugin.rpc.sendpay(route, payment_hash, payment_secret=payment_secret)

                    plugin.log('Checking time...')
                    running_for = int(time.time()) - start_ts

                    plugin.log('Executing waitsendpay...')
                    result = plugin.rpc.waitsendpay(payment_hash, max(retry_for - running_for, 0))
                    # time_sendpay += time.time() - time_start

                    if result.get('status') == "complete":
                        # rpc_result["stats"] = f"running_for:{int(time.time()) - start_ts}  time_sendpay:{time_sendpay}"
                        return cleanup(plugin, label, payload, rpc_result)

                except RpcError as e:

                    # time_sendpay += time.time() - time_start
                    # plugin.log(
                    # f"maxhops:{plugin.maxhopidx}  msatfactor:{plugin.msatfactoridx}  running_for:{int(time.time()) - start_ts} time_sendpay:{time_sendpay}",
                    # 'debug')

                    # check if we ran into the `rpc.waitsendpay` timeout
                    if e.method == "waitsendpay" and e.error.get('code') == 200:
                        raise RpcError("rebalance", payload, {'message': 'Timeout reached'})
    except Exception as e:
        return cleanup(plugin, label, payload, rpc_result, e)
    rpc_result = {'status': 'error', 'message': 'Timeout reached'}
    return cleanup(plugin, label, payload, rpc_result)


# endregion


@plugin.method('start_hide_seek')
def start_hide_seek(plugin):
    plugin.log('Starting hide and seek plugin...')
    # check if node is not participating in rebalancing already 
    if busyRebalancing.is_set():
        plugin.log("Ignore request to start hide and seek. Apperently I am already participating in rebalancing.")
        return
    else:
        busyRebalancing.set()
    plugin.log('Deciding if it makes sense to hide and seek...')
    # decide using nodes own objection function if rebalancing makes sense
    if not decide_to_participate(plugin):
        plugin.log("There is no need to participate in rebalancing for me.")
        return
    # start rebalancing daemon
    plugin.log('Starting the rebalancing deamon for hide and seek...')
    threading.Thread(name="initiator", target=prepare_hide_seek, args=(plugin,)).start()


# add response handler for global access
response_handler = ResponseHandler()


@plugin.init()
def init(options, configuration, plugin):
    plugin.log("Plugin hide_and_seek.py initialized.")
    return


plugin.add_notification_topic(MESSAGE_BUS_TOPIC)

plugin.run()
