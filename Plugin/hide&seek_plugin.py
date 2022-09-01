#!/usr/bin/env python3
from pyln.client import Plugin, Millisatoshi, RpcError
from threading import Thread, Lock
from datetime import timedelta
from functools import reduce
import time
import uuid

plugin = Plugin()

global_rebalancing_threshold = 50

plugin.rebalance_stop = False

@plugin.method("rebalance")
def hello(plugin, outgoing_scid, incoming_scid, msatoshi: Millisatoshi = None,
              retry_for: int = 60, maxfeepercent: float = 0.5,
              exemptfee: Millisatoshi = Millisatoshi(5000)):
    """Rebalancing of one particular channel of a node based on Hide & Seek protocol.

    White paper can be found here:
    https://arxiv.org/pdf/2110.08848.pdf
    """
    if msatoshi:
        msatoshi = Millisatoshi(msatoshi)
    retry_for = int(retry_for)
    maxfeepercent = float(maxfeepercent)
    exemptfee = Millisatoshi(exemptfee)
    payload = {
        "outgoing_scid": outgoing_scid,
        "incoming_scid": incoming_scid,
        "msatoshi": msatoshi,
        "retry_for": retry_for,
        "maxfeepercent": maxfeepercent,
        "exemptfee": exemptfee
    }

    # here are the rpc instructions
    my_node_id = plugin.rpc.getinfo().get('id')
    outgoing_node_id = peer_from_scid(plugin, outgoing_scid, my_node_id, payload)
    incoming_node_id = peer_from_scid(plugin, incoming_scid, my_node_id, payload)
    get_channel(plugin, payload, outgoing_node_id, outgoing_scid, True)
    get_channel(plugin, payload, incoming_node_id, incoming_scid, True)
    out_ours, out_total = amounts_from_scid(plugin, outgoing_scid)
    in_ours, in_total = amounts_from_scid(plugin, incoming_scid)

    # If amount was not given, calculate a suitable 50/50 rebalance amount
    if msatoshi is None:
        msatoshi = calc_optimal_amount(out_ours, out_total, in_ours, in_total, payload)
        plugin.log("Estimating optimal amount %s" % msatoshi)

    return s

@plugin.method("rebalanceall")
def rebalanceall(plugin: Plugin, min_amount: Millisatoshi = Millisatoshi("50000sat"), feeratio: float = 0.5):
    """Rebalancing of all unbalanced channels of a node based on Hide & Seek protocol.

    Default minimum rebalancable amount is _____ (50000) sat. Default feeratio = _._ (0.5), half of our node's default fee.
    To be economical, it tries to fix the liquidity cheaper than it can be ruined by transaction forwards.
    It may run for a long time (hours) in the background, but can be stopped with the rebalancestop method.

    White paper can be found here:
    https://arxiv.org/pdf/2110.08848.pdf
    """
    # some early checks before we start the async thread
    if plugin.mutex.locked():
        return {"message": "Rebalance is already running, this may take a while. To stop it use the cli method 'rebalancestop'."}
    channels = get_open_channels(plugin)
    if len(channels) <= 1:
        return {"message": "Error: Not enough open channels to rebalance anything"}
    our = sum(ch["to_us_msat"] for ch in channels)
    total = sum(ch["total_msat"] for ch in channels)
    min_amount = Millisatoshi(min_amount)
    if total - our < min_amount or our < min_amount:
        return {"message": "Error: Not enough liquidity to rebalance anything"}

    # param parsing ensure correct type
    plugin.feeratio = float(feeratio)
    plugin.min_amount = min_amount

    # run the job
    t = Thread(target=rebalanceall_thread, args=(plugin, ))
    t.start()
    return {"message": f"Rebalance started with min rebalancable amount: {plugin.min_amount}, feeratio: {plugin.feeratio}"}

@plugin.init()
def init(options, configuration, plugin):
    config = plugin.rpc.listconfigs()
    plugin.cltv_final = config.get("cltv-final")
    plugin.fee_base = Millisatoshi(config.get("fee-base"))
    plugin.fee_ppm = config.get("fee-per-satoshi")
    plugin.mutex = Lock()
    plugin.maxhops = int(options.get("rebalance-maxhops"))
    plugin.msatfactor = float(options.get("rebalance-msatfactor"))
    plugin.erringnodes = int(options.get("rebalance-erringnodes"))
    plugin.getroute = getroute_switch(options.get("rebalance-getroute"))
    plugin.rebalanceall_msg = None

    plugin.log(f"Plugin rebalance initialized with {plugin.fee_base} base / {plugin.fee_ppm} ppm fee  "
               f"cltv_final:{plugin.cltv_final}  "
               f"maxhops:{plugin.maxhops}  "
               f"msatfactor:{plugin.msatfactor}  "
               f"erringnodes:{plugin.erringnodes}  "
               f"getroute:{plugin.getroute.__name__}  ")

plugin.add_option('This is a first implementation of a rebalancing plugin based on Hide & Seek protocol. You can find all preliminaries and theoretical background at https://arxiv.org/pdf/2110.08848.pdf')
plugin.run()