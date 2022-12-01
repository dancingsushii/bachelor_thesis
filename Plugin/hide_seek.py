#!/usr/bin/env python3

from functools import reduce
from pyln.client import Plugin
import json, threading, random, os, subprocess, time, uuid, ast
import networkx as nx
from queue import Queue, Empty
from enum import Enum

import gurobipy as gp
import numpy as np
import scipy.sparse as sp
from gurobipy import GRB
from pyln.client import Plugin, RpcError, Millisatoshi

#region params
MESSAGE_BUS_TOPIC = 'hide_and_seek_message_bus'
REBALANCE_REQ_TIMEOUT_S = 60
REBALANCE_RES_TIMEOUT_S = 60*10
REBALANCE_REQ_SEARCH_DEPTH = 100
CYCLE_FLOW_REQUEST_TIMEOUT_S = 60
#endregion

plugin = Plugin()

#region networking and request dispatching

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
  # cycle flow information sharing #
  CYCLE_FLOW_REQ_MSG_HEX = 0x1045

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

  def __init__(self, initiator_id: str, participants: list, rejectors: list, search_depth: int, accum_depth: int) -> None:
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

  def open_request(self, request_id:str):
    """Prepare a queue for responses to your request."""
    if request_id not in self._responses:
      self._responses[request_id] = Queue() 

  def add_response(self, request_id:str, response: RebalanceMessage):
    """Add response to the response-queue of the request.
    
    Keyword arguments:\n
    request_id -- id of the corresponding request \n
    response -- response message.
    """
    self._responses[request_id].put(response)

  def get_response(self, request_id:str, timeout) -> RebalanceMessage:
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

  def is_response(self, response_id:str) -> bool:
    """Check if incoming message id is being awaited as a response to a happened request."""
    return response_id in self._responses

  def close_request(self, request_id:str):
    """Close request and process no responses to it anymore"""
    with self._remove_lock:
      if request_id not in self._responses:
        return
      request_responses = self._responses.pop(request_id)
      with request_responses.mutex:
        request_responses.queue.clear()

def send_message(plugin, peer_id: str, msgtype: HideSeekMsgCode, payload: str, msgid: int = None, expect_response: bool = False, response_handler: ResponseHandler = None) -> str:
  """Sends a message of ``msgtype`` type to ``peer_id`` with ``payload`` over RPC custommsg channels.\n
  ``msgid`` gets generated automatically if not provided <--> we provide msgid to the messages that are responding to existing request.\n
  If ``expect_response`` is set, ``response_handler`` **must** be provided. Otherwise it throws an exception!\n
  Returns ``hex()`` of the ``msgid``.\n"""
  if expect_response and response_handler is None:
    raise Exception(f"An error occured when sending message of type {msgtype.value} to {peer_id}: expect_response flag is set to True, but no response_handler provided to process the responses!\n")
  mtype = msgtype.value
  if msgid is None: msgid = random.randint(0, 2**64)
  msg = (mtype.to_bytes(2, 'big')
        + msgid.to_bytes(8, 'big')
        + bytes(payload, encoding='utf8'))
  request_id = hex(msgid)
  if expect_response:
    response_handler.open_request(request_id=request_id)
  res = plugin.rpc.call('sendcustommsg', {'node_id': peer_id, 'msg': msg.hex()})
  plugin.log(f'Request {request_id} sent, rpc call result {res}')
  return request_id

@plugin.async_hook('custommsg')
def on_custommsg(peer_id, payload, plugin, request, **kwargs):
  """Use custommsg hook to receive payload. 
  Pass messages of type ``HideSeekMsgCode`` to ``MESSAGE_BUS_TOPIC`` channel for processing."""
  pbytes  = bytes.fromhex(payload)
  mtype   = int.from_bytes(pbytes[:2], "big")
  if HideSeekMsgCode.has_value(mtype):
    msgid   = int.from_bytes(pbytes[2:10], "big")
    data    = pbytes[10:].decode()
    message = dict({ 'peer': peer_id, 'msgid': hex(msgid), 'mtype': mtype, 'data': data })
    plugin.notify(MESSAGE_BUS_TOPIC, json.dumps(message))
  return request.set_result({'result': 'continue'})

@plugin.subscribe(MESSAGE_BUS_TOPIC)
def on_newmessage(plugin, origin, payload, request, **kwargs):
  """Dispatches incoming requests to request handlers and incoming responses into ResponseHandler queues."""
  d = json.loads(payload)
  message = RebalanceMessage(d['peer'], d['msgid'], d['mtype'], d['data'])
  plugin.log(f"Received message {message.msg_id} of type {message.msg_type.value} from {message.peer}: {message.payload}")
  if response_handler.is_response(message.msg_id):
    response_handler.add_response(message.msg_id, message)
  elif message.msg_type == HideSeekMsgCode.REBALANCE_REQ_MSG_HEX:
    plugin.log('Received msg is a rebalance request.')
    handle_incoming_hide_seek_request(plugin, message)
  elif message.msg_type == HideSeekMsgCode.CYCLE_FLOW_REQ_MSG_HEX:
    plugin.log('Received msg is a cycle flow processing request.')
    handle_incoming_cycle_flow_processing_request(plugin, message)
  return
#endregion

#region handling and logic

def prepare_hide_seek(plugin, source_request: RebalanceMessage = None):
  """Hide & Seek data preparation handler.\n
  Differentiates between initiator and normal node by presence of the source_request:\n
  if source_request is present, the node received the request from another node. If not -> the node is the initiator."""
  plugin.log('Daemon initiating hide and seek...')
  own_node_id = get_node_id(plugin)
  rebalancing_graph = nx.DiGraph()
  this_node_is_initiator = source_request is None
  if this_node_is_initiator:
    initiator_id, participants, rejectors = own_node_id, [own_node_id], []
    search_depth = REBALANCE_REQ_SEARCH_DEPTH
    accum_depth = 0
  else:
    available_data = RebalanceRequestBody.fromString(source_request.payload)
    initiator_id, participants, rejectors = available_data.initiator_id, available_data.participants, available_data.rejectors
    participants.append(own_node_id) if own_node_id not in participants else participants
    accum_depth, search_depth = available_data.accum_depth, available_data.search_depth
    accum_depth += 1
  if accum_depth < search_depth:
    # send further requests and collect data from neighbors into the rebalancing_graph
    rebalancing_graph = collect_neighbors_replies(plugin, initiator_id, participants, rejectors, search_depth, accum_depth, rebalancing_graph)
  else: plugin.log(f'Required search depth achieved, I can resend my network data available to me.')
  plugin.log(f"All neighbors processed or search deptch achieved. Extending rebalancing graph with local data...")
  # when no neighbors are left, add locally available information to the rebalancing_graph
  plugin.log(f"Before extending the graph with local data, it contained {len(rebalancing_graph.nodes)} nodes and {len(rebalancing_graph.edges)} edges")
  rebalancing_graph = extend_rebalancing_graph(plugin, own_node_id, rebalancing_graph)
  if source_request is not None:
    # and send it back to the requestor
    plugin.log(f"Sending rebalancing graph back to the requestor...")
    send_rebalance_res(plugin, source_request, rebalancing_graph, participants, rejectors)
    return

  # we are at initiator. it has the rebalancing_graph.
  for edge in rebalancing_graph.edges:
    graph_for_LP = rebalancing_graph.get_edge_data(edge[0], edge[1])
    plugin.log(f"Edge between {edge[0]} and {edge[1]} has data: {graph_for_LP}")
    
  
  balance_updates = LP_global_rebalancing(plugin, rebalancing_graph)
  # Cycle decomposition
  cycle_flows = cycle_decomposition(balance_updates)
  # cycle_flows is a list containing cycle flows. Cycle flow is a list of triples [(alice, bob, 5), (bob, carol, 5), (carol, alice, 5)].
  cycle_flows_with_scids = []
  cycle_with_scid = []

  plugin.log(f"Cycle flows are: {cycle_flows}...")
  plugin.log(f"The total rebalancing graph has {len(rebalancing_graph.nodes)} nodes and {len(rebalancing_graph.edges)} edges... ")
  for cycle in cycle_flows:
    for edge in cycle:
      short_channel_id = rebalancing_graph[edge[0]][edge[1]]['short_channel_id']
      quadriple = (edge[0], edge[1], edge[2], short_channel_id)
      cycle_with_scid.append(quadriple)
  cycle_flows_with_scids.append(cycle_with_scid)
  plugin.log(f"The cycle flows with scid's are {cycle_flows_with_scids} ... ")

  for cycle_flow in cycle_flows:
    cycle_members = list(map(lambda cycle: cycle[0], cycle_flow))
    
    if own_node_id in cycle_members:
      plugin.log(f"Starting executing cycle {cycle_flow} from self.")
      htlc_creation_for_cycle(plugin, cycle_flow)

    else:
      # We pick one random member m_i from each of the cycles.
      random_member = random.choice(cycle_members)
      # we send them (m_i) cycle information for them (m_i) to call HTLC for cycle creation.
      try:
        plugin.log(f"Sending cycle information to {random_member}")
        cycle_information = str(cycle_flow)
        send_cycle_flow(plugin, random_member, payload=cycle_information)
        # wait for the response # TODO consider if we need to wait for request ACK!
        # request_ack = response_handler.get_response(request_id, 60) # TODO add timeout value to params region CYCLE_FLOW_REQUEST_TIMEOUT_S
        # except Empty:
        # plugin.log(f"Sending cycle information to {random_member} timed out!") 
        # TODO if request timed out, pick another member! Go back to last else:

      except Exception as e:
        error = str(e)
        plugin.log(f"Unknown error occured while sending cycle information to {random_member}. Exception: {error}") 

def collect_neighbors_replies(plugin, initiator_id: str, participants: list, rejectors: list, search_depth: int, accum_depth: int, rebalancing_graph: nx.DiGraph) -> nx.DiGraph:
  """Goes through neighbors, sends rebalance out requests and collects their local graphs into the common ``rebalancing_graph``"""
  neighbor_ids = get_neighbor_ids(plugin)
  for neighbor_id in neighbor_ids:
    if neighbor_id in participants or neighbor_id in rejectors:
      plugin.log(f"Neighbor {neighbor_id} already received invititation to participate.")
      continue
    plugin.log(f'Daemon picking {neighbor_id} to send rebalance out')
    try:
      request_body = RebalanceRequestBody(initiator_id, participants, rejectors, search_depth, accum_depth)
      # send rebalance_request to neighbor
      request_id = send_rebalance_out(plugin, neighbor_id, request_body)
      # wait until request times out or neighbor acknowledges
      response = response_handler.get_response(request_id, REBALANCE_REQ_TIMEOUT_S)
      # if peer rejected or response not recognized
      if response.msg_type == HideSeekMsgCode.REBALANCE_REJ_MSG_HEX:
        plugin.log(f'Peer {response.peer} responded with an REJ to request {request_id}')
        rejectors.append(neighbor_id)
        continue
      if response.msg_type != HideSeekMsgCode.REBALANCE_ACC_MSG_HEX:
        plugin.log(f'Received a response of unexpected type {response.msg_id} as a reply to request {request_id}')
        raise Exception(f'Received a response of unexpected type {response.msg_id} as a reply to request {request_id}')
        continue
      # ... peer accepted then
      plugin.log(f'Peer {response.peer} responded with an ACK to request {request_id}. Waiting for the graph data from the peer now...')
      # wait for neighbors rebalance response with rebalancing data
      response = response_handler.get_response(request_id, REBALANCE_RES_TIMEOUT_S)
      plugin.log(f"Response arrived! Parsing response to retrieve participants, rejectors and graph data.")
      response_payload = json.loads(response.payload)
      response_participants, response_rejectors, response_graph_data = response_payload["participants"], response_payload["rejectors"], response_payload["graph_data"] 
      # join our participants and response_participants
      for participant in response_participants: participants.append(participant) if participant not in participants else participants
      for rejector in response_rejectors: rejectors.append(rejector) if rejector not in rejectors else rejectors
      # collect graph data from the response
      plugin.log(f"Packing graph data into the rebalancing graph!")
      filename = "rebalancing_graph.gpickle.gz"
      if os.path.exists(filename): os.remove(filename)
      with open(filename, "wb") as f:
        f.write(bytes.fromhex(response_graph_data))
      plugin.log(f"Graph data from host {neighbor_id} written into the file {filename}. Reading graph data from there...")
      peers_rebalancing_graph: nx.DiGraph = nx.read_gpickle(filename)
      plugin.log(f"Delivered graph contains {len(peers_rebalancing_graph.nodes)} nodes and {len(peers_rebalancing_graph.edges)} edges")
      # merge the existing rebalancing_graph with TODO read this methods docks after accomplishing the correct data collection
      rebalancing_graph = nx.compose(rebalancing_graph, peers_rebalancing_graph)
      plugin.log(f"Composed graph contains {len(rebalancing_graph.nodes)} nodes and {len(rebalancing_graph.edges)} edges")
  # return collected graph data
    except Empty:
      # handle timeout
      plugin.log(f'Request {request_id} timed out')
    except Exception as e:
      # handle other errors
      exception = str(e)
      plugin.log(f'An unexpected error occured during request handling of request {request_id}. Exception information: \n{exception}\n')
  
  return rebalancing_graph

def handle_incoming_hide_seek_request(plugin, request:RebalanceMessage):
  # decide using nodes own objection function if rebalancing makes sense
  if not decide_to_participate(plugin):
    plugin.log("There is no need to participate in rebalancing for me.")
    send_rebalance_rej(plugin, request.peer)
    return
  # simply send ACK for now, need to check if ResponseHandler works properly
  send_rebalance_ack(plugin, request.peer, request.msg_id)
  plugin.log(f'Request ACK sent to {request.peer}')
  # start rebalancing deamon
  plugin.log('Starting the rebalancing deamon for hide and seek...')
  threading.Thread(name="handler", target=prepare_hide_seek, args=(plugin, request)).start()

def handle_incoming_cycle_flow_processing_request(plugin, request:RebalanceMessage):
  # TODO add ACK logic if we want to get ACK for received cycle in order to retransmit the cycle to another random peer
  plugin.log(f"Cycle flow: {request.payload} received... ")
  deserialized_cycle = ast.literal_eval(request.payload)
  htlc_creation_for_cycle(plugin, deserialized_cycle)

def send_rebalance_out(plugin, peer_id, request_body: RebalanceRequestBody) -> int:
  """Sends rebalance out request to given peer_id."""
  payload = request_body.toString()
  msgtype = HideSeekMsgCode.REBALANCE_REQ_MSG_HEX
  request_id = send_message(plugin, peer_id, msgtype, payload, expect_response = True, response_handler = response_handler)
  return request_id

def send_rebalance_rej(plugin, peer_id, request_id):
  """Sends rebalance REJ response to given peer_id."""
  own_id = get_node_id(plugin)
  payload = f"REJ from {own_id}"
  msgtype = HideSeekMsgCode.REBALANCE_REJ_MSG_HEX
  msgid   = int(request_id, base=0)
  send_message(plugin, peer_id, msgtype, payload, msgid)

def send_rebalance_ack(plugin, peer_id, request_id):
  """Sends rebalance ACK response to given peer_id."""
  own_id = get_node_id(plugin)
  payload = f"ACK from {own_id}"
  msgtype = HideSeekMsgCode.REBALANCE_ACC_MSG_HEX
  msgid   = int(request_id, base=0)
  send_message(plugin, peer_id, msgtype, payload, msgid)

def send_rebalance_res(plugin, source_request: RebalanceMessage, rebalancing_graph: nx.DiGraph, participants: list, rejectors: list):
  """Pickles and compresses ``rebalancing_graph`` to send it as a reply to the ``source_request``."""
  # pickle and compress rebalancing graph
  filename = f"rebalancing_graph.gpickle.gz"
  if os.path.exists(filename): os.remove(filename)
  nx.write_gpickle(rebalancing_graph, filename)
  plugin.log(f"Complete graph of {len(rebalancing_graph.nodes)} nodes and {len(rebalancing_graph.edges)} edges created and written into {filename}")
  # write it into the rebalancing response
  with open(filename, "rb") as f:
    graph_data = (f.read()).hex()
  payload_body = {
    "participants": participants,
    "rejectors": rejectors,
    "graph_data": graph_data
  }
  payload = json.dumps(payload_body)
  msgtype = HideSeekMsgCode.REBALANCE_RES_MSG_HEX
  msgid = int(source_request.msg_id, base=0)
  request_id = send_message(plugin, source_request.peer, msgtype, payload, msgid)
  return request_id

def send_cycle_flow(plugin, peer_id, payload):
  """Sends cycle flow request to given peer_id."""
  plugin.log(f"Send cycle flow: {payload} to peer: {peer_id}...")
  # expect_response=True,  response_handler=response_handler deleted since we don't use ACK's now
  send_message(plugin, peer_id, HideSeekMsgCode.CYCLE_FLOW_REQ_MSG_HEX, payload)

# #endregion

# #region graph logic

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

def extend_rebalancing_graph_with_local_data(plugin, own_id, rebalancing_graph: nx.DiGraph) -> nx.DiGraph:
  """Extends the rebalancing graph with the required local graph information.\n
  Edges get added with ``initial_balance`` values in msat that mean: for A --> B edge ``initial_balance`` = how much msat can A send to B.\n
  channel["inflight"]["our_funding_msat"] contains initial balance attribute in listpeers RPC. Initial balance is not required for hide and seek."""
  
  peers = get_peers(plugin)
  peer_ids = list(map(lambda peer: peer["id"], peers))
  plugin.log(f"Collecting local rebalancing graph... collected peer_ids {peer_ids}")

  for peer in peers:
    
    peer_id, channel = peer["id"], peer["channels"][0]

    # These two are for dual funding functionality
    # total_balance = int(channel["inflight"]["total_funding_msat"].replace("msat", ""))
    # own_initial_balance = int(channel["inflight"]["our_funding_msat"].replace("msat", ""))
  
    total_balance = int(channel["total_msat"])
    own_initial_balance = int(channel["to_us_msat"])
    remote_initial_balance = total_balance - own_initial_balance
    scid = channel["short_channel_id"]

    half_of_capacity = total_balance/2
    if own_initial_balance > half_of_capacity:
      own_flow_bound, remote_flow_bound = own_initial_balance - half_of_capacity, 0
      own_obj_func_coef, remote_obj_func_coef = 1, 0
    else:
      own_flow_bound, remote_flow_bound = 0, remote_initial_balance - half_of_capacity
      own_obj_func_coef, remote_obj_func_coef = 0, 1


    if not rebalancing_graph.has_edge(own_id, peer_id):
      rebalancing_graph.add_edge(own_id, peer_id, flow_bound = own_flow_bound, short_channel_id = scid, objective_function_coefficient = own_obj_func_coef)
    if not rebalancing_graph.has_edge(peer_id, own_id):
      rebalancing_graph.add_edge(peer_id, own_id, flow_bound = remote_flow_bound, short_channel_id = scid, objective_function_coefficient = remote_obj_func_coef)    

  return rebalancing_graph

def extend_rebalancing_graph(plugin, own_id, rebalancing_graph: nx.DiGraph) -> nx.DiGraph:
  """Extends the rebalancing graph with data required for rebalancing: ``flow_bound``, ``objective function coefficient``, ``scid``."""
  
  peers = get_peers(plugin)
  peer_ids = list(map(lambda peer: peer["id"], peers))
  plugin.log(f"Collecting local rebalancing graph... collected peer_ids {peer_ids}")

  for peer in peers:
    
    peer_id, channel = peer["id"], peer["channels"][0]

    # These two are for dual funding functionality
    # total_balance = int(channel["inflight"]["total_funding_msat"].replace("msat", ""))
    # own_initial_balance = int(channel["inflight"]["our_funding_msat"].replace("msat", ""))

    total_balance = int(channel["total_msat"])
    own_initial_balance = int(channel["to_us_msat"])
    remote_initial_balance = total_balance - own_initial_balance
    scid = channel["short_channel_id"]
    
    half_of_capacity = total_balance/2
    if own_initial_balance > half_of_capacity:
      own_flow_bound, remote_flow_bound = own_initial_balance - half_of_capacity, 0
      own_obj_func_coef, remote_obj_func_coef = 1, 0
    else:
      own_flow_bound, remote_flow_bound = 0, remote_initial_balance - half_of_capacity
      own_obj_func_coef, remote_obj_func_coef = 0, 1

    if not rebalancing_graph.has_edge(own_id, peer_id):
      rebalancing_graph.add_edge(own_id, peer_id, flow_bound = own_flow_bound, short_channel_id = scid, objective_function_coefficient = own_obj_func_coef)
    if not rebalancing_graph.has_edge(peer_id, own_id):
      rebalancing_graph.add_edge(peer_id, own_id, flow_bound = remote_flow_bound, short_channel_id = scid, objective_function_coefficient = remote_obj_func_coef) 

  return rebalancing_graph
  
# #endregion

# # region MPC blackbox

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

def LP_global_rebalancing(plugin, rebalancing_graph) -> list:
    try:
        n = rebalancing_graph.number_of_nodes()
        m = rebalancing_graph.number_of_edges()
        list_of_nodes = list(rebalancing_graph.nodes)
        list_of_edges = list(rebalancing_graph.edges)

        # Create a new model, variables and set an objective
        model = gp.Model("rebalancing-LP")
        x = model.addMVar(shape=m, vtype=GRB.CONTINUOUS, name="x")
        obj = np.zeros(m, dtype=float)

        plugin.log(f"gp.Model initialized")

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

        plugin.log(f'done with constraint 1')

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
                plugin.log(f'Row number is {row}')

            for edge in rebalancing_graph.in_edges(u):
                edge_index = list_of_edges.index(edge)

                # ineq 2a: - \sum_{in edges} f(v,u)
                append_to_A(-1, 2 * m + i, edge_index, data, row, col)

                # ineq 2b: \sum_{in edges} f(v,u)
                append_to_A(1, 2 * m + n + i, edge_index, data, row, col)

        plugin.log('done with constraint 2')

        A_num_of_rows = 2 * m + 2 * n
        A_num_of_columns = m

        A = sp.csr_matrix((data, (row, col)), shape=(A_num_of_rows, A_num_of_columns))

        # Add constraints and optimize model
        plugin.log(f'A.data is {A.data}')
        plugin.log(f'RHS is {rhs}')
        model.addConstr(A @ x <= rhs, name="matrix form constraints")
        model.optimize()

        try:
            plugin.log(f'{x.X}')
            plugin.log(f'Obj: {model.objVal}')

            flows = x.X
            plugin.log(f'Flows are: {flows}')
        except:
            # infeasible model, set all flows to zero
            plugin.log('model is infeasible, setting all flows to zero')
            flows = list(np.zeros(m, dtype=int))

        balance_updates = []

        # flow updates
        for edge_index in range(m):
            u, v = list_of_edges[edge_index]
            balance_updates.append((u, v, int(flows[edge_index])))
            plugin.log(f'Balance updates is constructing {balance_updates}')

        return balance_updates

    except gp.GurobiError as e:
        plugin.log('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        plugin.log('Encountered an attribute error')

def cycle_decomposition(balance_updates) -> list:
    # Step 6: Cycle decomposition on MPC delegate
    # TODO Enhancement add case where the model received from 1LP is infeasible meaning there will be nothing to route
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

#endregion

# region HTLC creation for cycles

def scid_from_peer(plugin, peer_id):
    """Returns ``scid`` for the channel with a given peer"""
    peers = plugin.rpc.listpeers(peer_id).get('peers')
    filter_strategy = lambda peer: peer["id"] == peer_id and "channels" in peer and len(peer["channels"]) > 0 \
                                   and peer["channels"][0]["state"] == "CHANNELD_NORMAL" and peer["connected"] == True
    filtered_peers = list(filter(filter_strategy, peers))
    target_peer = filtered_peers[0]
    target_peer_channel = target_peer["channels"][0]
    return target_peer_channel["short_channel_id"]


def peer_from_scid(plugin, short_channel_id, my_node_id, payload):
    channels = plugin.rpc.listchannels(short_channel_id).get('channels')
    for ch in channels:
        if ch['source'] == my_node_id:
            return ch['destination']
    raise RpcError("rebalance", payload, {'message': 'Cannot find peer for channel: ' + short_channel_id})


def amounts_from_scid(scid):
    channels = plugin.rpc.listfunds().get('channels')
    channel = next(c for c in channels if c.get('short_channel_id') == scid)
    our_msat = Millisatoshi(channel['our_amount_msat'])
    total_msat = Millisatoshi(channel['amount_msat'])
    return our_msat, total_msat


def get_node_alias(node_id):
    node = plugin.rpc.listnodes(node_id)['nodes']
    s = ""
    if len(node) != 0 and 'alias' in node[0]:
        s += node[0]['alias']
    else:
        s += node_id[0:7]
    return s


def get_channel(plugin, payload, peer_id, scid, check_state: bool = False):
    peer = plugin.rpc.listpeers(peer_id)['peers'][0]
    #peer = list_peers(peer_id).get('peers')[0]
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


def sort_cycle(current: str, cycle: list, result_cycle: list) -> list:
  if len(cycle) == 0:
    return result_cycle
  current_tuple = next(filter(lambda element: element[0] == current, cycle))
  result_cycle.append(current_tuple)
  current = current_tuple[1]
  cycle.remove(current_tuple)
  return sort_cycle(current, cycle, result_cycle)
  

def htlc_creation_for_cycle(plugin, cycle: list, retry_for: int = 60):
    """Method for cyclic creation of HTLC.

    Is initiated randomly on one of the cycle participant after 1LP calculation.
    """
    plugin.log('Starting htlc_creation_for_cycles method...')

    my_node_id = get_node_id(plugin)

    if cycle[0][0] != my_node_id:
      plugin.log(f'The starting node in cycle aren`t us, we are {my_node_id} and the starting is {cycle[0][0]}.')
      result_cycle = []
      cycle = sort_cycle(my_node_id, cycle, result_cycle)

    msatoshi = cycle[0][2]
    if msatoshi:
        msatoshi = Millisatoshi(msatoshi)
    retry_for = int(retry_for)
    plugin.log(f'The amount of funds flowing is {msatoshi} and retry for is set to {retry_for}')

    outgoing_scid = scid_from_peer(plugin, cycle[0][1])
    incoming_scid = scid_from_peer(plugin, cycle[-1][0])
    payload = {
        "outgoing_scid": outgoing_scid,
        "incoming_scid": incoming_scid,
        "msatoshi": msatoshi,
        "retry_for": retry_for,
    }

    my_peers = plugin.rpc.listpeers()['peers']
    plugin.log(f"My node info is: {my_node_id} and my peers are {my_peers}")

    outgoing_node_id = peer_from_scid(plugin, outgoing_scid, my_node_id, payload)
    incoming_node_id = peer_from_scid(plugin, incoming_scid, my_node_id, payload)
    route_out = {'id': outgoing_node_id, 'channel': outgoing_scid}
    route_in = {'id': my_node_id, 'channel': incoming_scid}

    plugin.log(f"The overall payload is {payload}")

    get_channel(plugin, payload, outgoing_node_id, outgoing_scid, True)
    get_channel(plugin, payload, incoming_node_id, incoming_scid, True)
    out_ours, out_total = amounts_from_scid(outgoing_scid)
    in_ours, in_total = amounts_from_scid(incoming_scid)

    if msatoshi > out_ours or msatoshi > in_total - in_ours:
      raise RpcError("rebalance", payload, {'message': 'Channel capacities too low'})

    plugin.log(f"Starting rebalance out_scid:{outgoing_scid} in_scid:{incoming_scid} amount:{msatoshi}", 'debug')
    plugin.log('Creating timelock for cycle...')
    route_out = {'id': outgoing_node_id, 'channel': outgoing_scid, 'direction': int(not my_node_id < outgoing_node_id)}
    route_in = {'id': my_node_id, 'channel': incoming_scid, 'direction': int(not incoming_node_id < my_node_id)}
    start_ts = int(time.time())
    label = "Rebalance-" + str(uuid.uuid4())
    description = "%s to %s" % (outgoing_scid, incoming_scid)
    plugin.log(f'Start time is {start_ts} with the label {label} and description {description}...')

    plugin.log('Creating invoice for the whole cycle...')
    invoice = plugin.rpc.invoice(msatoshi, label, description, retry_for + 60)

    plugin.log('Creating payment_secret for the whole cycle...')
    payment_secret = invoice.get('payment_secret')

    plugin.log('Creating a hash for this secret for the whole cycle...')
    payment_hash = invoice['payment_hash']

    rpc_result = None
    excludes = [my_node_id]  
    nodes = {}  
    count = 0
    count_sendpay = 0
    time_getroute = 0
    time_sendpay = 0

    try:
        while int(time.time()) - start_ts < retry_for:

            plugin.log(f"Setting the timer for {time.time() - start_ts} being less then {retry_for}")
            plugin.log('Trying to construct the midroute from cycle...')

            # edges_mid = (carol_id, bob_id, 25000)
            # OR edges_mid = (carol_id, bob_id, 25000),(bob_id, emma_id, 25000),(emma_id, dave_id, 25000)
            edges_mid = cycle[1:-1]
            route_mid = []

            for edge in edges_mid:
              plugin.log(f"Casting amount of satoshi in satoshi for {edge}...")
              # TODO Check if scid's read correctly, Here we have a problem, fix this
              hop = {'id': edge[1], 'channel': edge[3],'msatoshi': edge[2], 'amount_msat': Millisatoshi(edge[2])}
              route_mid.append(hop)
            
            plugin.log(f"The intermediary edges are {edges_mid} and a mid route is {route_mid}")
            plugin.log(f'Parse the edges in edges_mid to route... rpc_result is: {rpc_result}')
            
            route = [route_out] + route_mid + [route_in]

            rpc_result = {
                "sent": msatoshi,
                "received": msatoshi,
                "hops": len(cycle),
                "outgoing_scid": outgoing_scid,
                "incoming_scid": incoming_scid,
                "status": "complete",
                "message": f"{msatoshi} sent over {len(cycle)} hops to rebalance {msatoshi}",
            }

            midroute_str = reduce(lambda x,y: x + " -> " + y, map(lambda r: get_node_alias(r['id']), route_mid))
            full_route_str = "%s -> %s -> %s -> %s" % (get_node_alias(my_node_id), get_node_alias(outgoing_node_id), midroute_str, get_node_alias(my_node_id))
            plugin.log("%d hops and %s fees for %s along route: %s" % (len(cycle), msatoshi.to_satoshi_str(), full_route_str))

            time_start = time.time()
            count_sendpay += 1

            try:
                plugin.log('Executing sendpay...')
                plugin.rpc.sendpay(route, payment_hash, payment_secret=payment_secret)
                running_for = int(time.time()) - start_ts
                result = plugin.rpc.waitsendpay(payment_hash, max(retry_for - running_for, 0))
                time_sendpay += time.time() - time_start
                if result.get('status') == "complete":
                    rpc_result["stats"] = f"running_for:{int(time.time()) - start_ts}  count_getroute:{count}  time_getroute:{time_getroute}  time_getroute_avg:{time_getroute / count}  count_sendpay:{count_sendpay}  time_sendpay:{time_sendpay}  time_sendpay_avg:{time_sendpay / count_sendpay}"
                    return cleanup(label, payload, rpc_result)

            except RpcError as e:
                time_sendpay += time.time() - time_start
                plugin.log(f"maxhops:{plugin.maxhopidx}  msatfactor:{plugin.msatfactoridx}  running_for:{int(time.time()) - start_ts}  count_getroute:{count}  time_getroute:{time_getroute}  time_getroute_avg:{time_getroute / count}  count_sendpay:{count_sendpay}  time_sendpay:{time_sendpay}  time_sendpay_avg:{time_sendpay / count_sendpay}", 'debug')
                # plugin.log(f"RpcError: {str(e)}", 'debug')
                # check if we ran into the `rpc.waitsendpay` timeout
                if e.method == "waitsendpay" and e.error.get('code') == 200:
                    raise RpcError("rebalance", payload, {'message': 'Timeout reached'})
                # check if we have problems with our own channels
                erring_node = e.error.get('data', {}).get('erring_node')
                erring_channel = e.error.get('data', {}).get('erring_channel')
                erring_direction = e.error.get('data', {}).get('erring_direction')
                if erring_channel == incoming_scid:
                    raise RpcError("rebalance", payload, {'message': 'Error with incoming channel'})
                if erring_channel == outgoing_scid:
                    raise RpcError("rebalance", payload, {'message': 'Error with outgoing channel'})
                # exclude other erroring channels
                if erring_channel is not None and erring_direction is not None:
                    excludes.append(erring_channel + '/' + str(erring_direction))
                # count and exclude nodes that produce a lot of errors
                if erring_node and plugin.erringnodes > 0:
                    if nodes.get(erring_node) is None:
                        nodes[erring_node] = 0
                    nodes[erring_node] += 1
                    if nodes[erring_node] >= plugin.erringnodes:
                        excludes.append(erring_node)
    except Exception as e:
        return cleanup(plugin, label, payload, rpc_result, e)
    rpc_result = {'status': 'error', 'message': 'Timeout reached'}
    return cleanup(plugin, label, payload, rpc_result)

#endregion

# add response handler for global access
response_handler = ResponseHandler()

@plugin.init()
def init(options, configuration, plugin):
    plugin.log("Plugin hide_and_seek.py initialized.")
    return


@plugin.method('start_hide_seek')
def start_hide_seek(plugin):
    plugin.log('Starting hide and seek plugin...')
    # decide using nodes own objection function if rebalancing makes sense
    if not decide_to_participate(plugin):
      plugin.log("There is no need to participate in rebalancing for me.")
      return
    # start rebalancing daemon
    plugin.log('Starting the rebalancing deamon for hide and seek...')
    threading.Thread(name="initiator", target=prepare_hide_seek, args=(plugin, )).start()

plugin.add_notification_topic(MESSAGE_BUS_TOPIC)

plugin.run()