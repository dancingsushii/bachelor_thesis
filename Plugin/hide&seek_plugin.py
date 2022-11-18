#!/usr/bin/env python3
import time
import uuid

from pyln.client import Plugin, Millisatoshi
import json, threading, random
import networkx as nx
from queue import Queue, Empty
from enum import Enum
from gurobipy import GRB
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# That is how I understand it now:
# Step 1: one node calls rebalanceall and starts to collect information about neighbours via sending them custom message: https://lightning.readthedocs.io/lightning-sendcustommsg.7.html
# Step 2: Following objective 50/50 function decisionMaker,
    #  bcs we rely on sendcustommsg each of the participants discovers information about other nodes not in recursive way but in TODO way
# Step 3: All data are packed and then is put together for input in *mocked MPC* for node initiator
# Step 4: Initiator node receives data as a graph node ids, channels (v, u, m(v, u))
# Step 5: Executes LP on that graph (need a license be deployed)
# Step 6: Cycle decomposition
# Step 7: Send instructions to other nodes 7.1 iterate through transaction list aka u -> v n satoshi via x channel
#                                         OR 7.2 (big amount of nodes causes flooding)

# Step 8: Try to build interface for execution of received transaction on each node
# Step 9: Rebalancing is done after some time when all transactions are executed


# Parameters to consider:
    # 1) total amount of generated transactions
    # 2) time needed for generated transactions

# Metrics:
    # success ratio = succ_trans / total_num_trans
    # success volume ratio = succ_vol / total_volume
    # duration time

### TODO see the final version of test.py for some insights.
### TODO for now implement it only with a very simple waiting logic
### TODO write the networking part of the plugin, assuming that every node is willing to participate (use it as a separate function returning always true)
### TODO by that assume that nodes will always allow 50% of their current channel balance to share for rebalancing purposes
### TODO log the retrieved network structure somehow. Like this node has those connections and this node has those.

### TODO why not reform all the dict as request/response containers and use dto classes instead?
### TODO will it work if we put code into different files?

# region params
MESSAGE_BUS_TOPIC = 'hide_and_seek_message_bus'
REBALANCE_REQ_TIMEOUT_S = 60
REBALANCE_RES_TIMEOUT_S = 60 * 10
REBALANCE_REQ_SEARCH_DEPTH = 5
# endregion

plugin = Plugin()
busyRebalancing = threading.Event()


# region networking

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
    """Dispatches incoming requests and responses."""
    d = json.loads(payload)
    message = RebalanceMessage(d['peer'], d['msgid'], d['mtype'], d['data'])
    plugin.log(
        f"Received message {message.msg_id} of type {message.msg_type.value} from {message.peer}: {message.payload}")
    if response_handler.is_response(message.msg_id):
        plugin.log(f"Received message is a response to existing request {message.msg_id}")
        response_handler.add_response(message.msg_id, message)
    elif message.msg_type == HideSeekMsgCode.REBALANCE_REQ_MSG_HEX:
        plugin.log('Received msg is a rebalance request.')
        dispatch_hide_seek_request(plugin, message)
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


def get_node_id(plugin):
    return plugin.rpc.getinfo()["id"]


def get_neighbor_ids(plugin):
    node_id = get_node_id(plugin)
    # get information on your connected peers
    neighbors = plugin.rpc.call('listchannels', {'source': node_id})
    # retrieve their ids
    neighbor_ids = map(lambda channel: channel["destination"], neighbors["channels"])
    return neighbor_ids


# endregion

# region commands transmission and handling

def follow_hide_seek(plugin, source_request: RebalanceMessage):
    """This method will be used to understand how to parametrize intiate_hide_seek and merged into it."""
    plugin.log('Daemon initiating hide and seek...')
    neighbor_ids = get_neighbor_ids(plugin)
    available_data = RebalanceRequestBody.fromString(source_request.payload)
    participants = available_data.participants
    rejectors = available_data.rejectors
    search_depth = available_data.search_depth
    accum_depth = available_data.accum_depth + 1
    if accum_depth >= search_depth:
        plugin.log(f'Required search depth achieved, I can resend my network data available to me.')
        # collect local graph data to share TODO the following code is just a CONCEPT!
        # compress and prepare it for transmission
        rebalancing_graph = collect_local_rebalancing_graph(plugin)  # probably a str or byte[]
        # send it back as a response
        return
    collect_neighbors_replies(plugin, neighbor_ids, participants, rejectors)
    # when no neighbors are left, we can assemble available information and return it back


def collect_neighbors_replies(plugin, neighbor_ids, participants, rejectors):
    for neighbor_id in neighbor_ids:
        if neighbor_id in participants or neighbor_id in rejectors:
            plugin.log(f"Neighbor {neighbor_id} already received invititation to participate.")
            continue
        plugin.log('Daemon picking {} to send rebalance out'.format(neighbor_id))
        try:
            request_body = RebalanceRequestBody(get_node_id(plugin), participants, rejectors,
                                                REBALANCE_REQ_SEARCH_DEPTH, 0)
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
            plugin.log('Peer {} responded with an ACK to request {}'.format(response.peer, request_id))
            # wait for neighbors rebalance response with rebalancing data
            response = response_handler.get_response(request_id, REBALANCE_RES_TIMEOUT_S)
            # dispatch data - TODO collect delivered nodes data together somehow

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


def initiate_hide_seek(plugin):
    """Rebalancing ``initiator thread`` will be locked in this function."""
    plugin.log('Daemon initiating hide and seek...')
    neighbor_ids = get_neighbor_ids(plugin)
    # TODO need to send search depth and timeouts too
    # TODO rewrite initiate_hide_seek to be parametrized
    participants = []
    rejectors = []
    collect_neighbors_replies(plugin, neighbor_ids, participants, rejectors)


def dispatch_hide_seek_request(plugin, request: RebalanceMessage):
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
    threading.Thread(name="initiator", target=follow_hide_seek, args=(plugin,)).start()


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


# endregion

# region graph logic

def decide_to_participate(plugin) -> bool:
    return True


def collect_local_rebalancing_graph(plugin, neighbor_ids):
    """Collects all the required local rebalancing graph information,
    transforms it into a networkx graph, pickles and compresses it to send to the delegate."""
    for neighbor_id in neighbor_ids:
        print('afawif;asljf')  # TODO


def assemble_responses(plugin, responses, own_graph):
    """Collects and assembles responses (inlc. own graph) into a pickled and compressed networkx graph"""
    plugin.log("Start screaming around if the size of the pickle file size is larger than 64k bytes size limitation")
    # TODO


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
def htlc_creation_for_cycles(cycles):

    for cycle in cycles:
        # Assumption: we are executing all the cycle from the node initiator
        # In the paper it happens randomly between all of the nodes in the cycles and it has to happen this way in reality

        # The requirement for payment_secret coincided with its addition to the invoice output.
        invoice = plugin.rpc.invoice()
        payment_secret = invoice.get('payment_secret')
        # timelock tc ←− len(c)
        start_ts = int(time.time())

        # uc chooses random secret rc and creates hash hc = H(rc)
        # for ec = (u, v) ∈ c starting from uc do
        for edge in cycle:
            # u creates HTLC(u, v, wc, hc, tc)
            msatoshi = Millisatoshi(cycle[3])
            label = "Rebalance-" + str(uuid.uuid4())
            description = "%s to %s" % (edge[1]['outgoing_scid'], cycle[-1]['outgoing_scid'])
            invoice = plugin.rpc.invoice(msatoshi, label, description)
            payment_hash = invoice['payment_hash']

                # decrement tc by 1

        return invoice


def wait_for_htlcs(plugin, failed_channels: list, scids: list = None):
    # HTLC settlement helper
    # taken and modified from pyln-testing/pyln/testing/utils.py
    result = True
    peers = plugin.rpc.listpeers()['peers']
    for p, peer in enumerate(peers):
        if 'channels' in peer:
            for c, channel in enumerate(peer['channels']):
                if scids is not None and channel.get('short_channel_id') not in scids:
                    continue
                if channel.get('short_channel_id') in failed_channels:
                    result = False
                    continue
                if 'htlcs' in channel:
                    if not wait_for(lambda: len(plugin.rpc.listpeers()['peers'][p]['channels'][c]['htlcs']) == 0):
                        failed_channels.append(channel.get('short_channel_id'))
                        plugin.log(f"Timeout while waiting for htlc settlement in channel {channel.get('short_channel_id')}")
                        result = False
    return result


def wait_for(success, timeout: int = 60):
    # cyclical lambda helper
    # taken and modified from pyln-testing/pyln/testing/utils.py
    start_time = time.time()
    interval = 0.25
    while not success():
        time_left = start_time + timeout - time.time()
        if time_left <= 0:
            return False
        time.sleep(min(interval, time_left))
        interval *= 2
        if interval > 5:
            interval = 5
    return True

# endregion



@plugin.method('start_hide_seek')
def start_hide_seek(plugin):
    plugin.log('Starting hide and seek plugin...')
    # check if node is not participating in rebalancing already
    if busyRebalancing.is_set():
        plugin.log("Ignore request to start hide and seek. Apparently I am already participating in rebalancing.")
        return
    else:
        busyRebalancing.set()
    plugin.log('Deciding if it makes sense to hide and seek...')
    # decide using nodes own objection function if rebalancing makes sense
    if not decide_to_participate(plugin):
        plugin.log("There is no need to participate in rebalancing for me.")
        return
    # start rebalancing daemon
    plugin.log('Starting the rebalancing daemon for hide and seek...')
    threading.Thread(name="initiator", target=initiate_hide_seek, args=(plugin,)).start()


# add response handler for global access
response_handler = ResponseHandler()


@plugin.init()
def init(options, configuration, plugin):
    plugin.log("Plugin hide_and_seek.py initialized.")
    return


plugin.add_notification_topic(MESSAGE_BUS_TOPIC)

# TODO add all attributes from params region as plugin options

plugin.run()