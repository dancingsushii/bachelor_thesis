#!/usr/bin/env python3
from pyln.client import Plugin, Millisatoshi, RpcError
from threading import Thread, Lock
from datetime import timedelta
from functools import reduce
import time
import uuid

plugin = Plugin()
plugin.rebalance_stop = False

# Hide & Seek global variables
global_rebalancing_threshold = 50


# Plugin initialization
@plugin.init()
def init(options, configuration, plugin):
    plugin.log("Hide & Seek rebalancing plugin initialized")

    # Collect all necessary information about current node
    config = plugin.rpc.listconfigs()
    plugin.cltv_final = config.get("cltv-final")
    plugin.fee_base = Millisatoshi(config.get("fee-base"))
    plugin.fee_ppm = config.get("fee-per-satoshi")
    plugin.mutex = Lock()

    # Options are configurable parameters for rebalancing plugin e.g. max hops, msatfactor etc.
    # plugin.maxhops = int(options.get("rebalance-maxhops"))
    # plugin.msatfactor = float(options.get("rebalance-msatfactor"))
    # plugin.erringnodes = int(options.get("rebalance-erringnodes"))
    # plugin.getroute = getroute_switch(options.get("rebalance-getroute"))
    # plugin.rebalanceall_msg = None

    plugin.log(f"Plugin rebalance initialized with {plugin.fee_base} base / {plugin.fee_ppm} ppm fee  "
               f"cltv_final:{plugin.cltv_final}  "
               # f"maxhops:{plugin.maxhops}  "
               # f"msatfactor:{plugin.msatfactor}  "
               # f"erringnodes:{plugin.erringnodes}  "
               # f"getroute:{plugin.getroute.__name__}  "
               )

# TODO
@plugin.method("rebalanceall")
def rebalanceall(plugin: Plugin, min_amount: Millisatoshi = Millisatoshi("50000sat"), feeratio: float = 0.5):
    """Rebalancing of all unbalanced channels of a node based on Hide & Seek protocol.

    Default minimum rebalancable amount is _____ (50000) sat. Default feeratio = _._ (0.5), half of our node's default fee.
    To be economical, it tries to fix the liquidity cheaper than it can be ruined by transaction forwards.
    It may run for a long time (hours) in the background, but can be stopped with the rebalancestop method.

    White paper can be found here:
    https://arxiv.org/pdf/2110.08848.pdf
    """

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
    t = Thread(target=rebalanceall_thread, args=(plugin,))
    t.start()
    return {
        "message": f"Rebalance started with min rebalancable amount: {plugin.min_amount}, feeratio: {plugin.feeratio}"}

# TODO
@plugin.method("rebalance")
def rebalance(plugin, outgoing_scid, incoming_scid, msatoshi: Millisatoshi = None,
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

# TODO
@plugin.method("rebalancestop")
def rebalancestop(plugin: Plugin):
    """It stops the ongoing rebalanceall.
    """
    if not plugin.mutex.locked():
        if plugin.rebalanceall_msg is None:
            return {"message": "No rebalance is running, nothing to stop."}
        return {"message": f"No rebalance is running, nothing to stop. "
                           f"Last 'rebalanceall' gave: {plugin.rebalanceall_msg}"}
    plugin.rebalance_stop = True
    plugin.mutex.acquire(blocking=True)
    plugin.rebalance_stop = False
    plugin.mutex.release()
    return {"message": plugin.rebalanceall_msg}

# TODO
@plugin.method("rebalancereport")
def rebalancereport(plugin: Plugin):
    """Show information about rebalance
    """
    res = {}
    res["rebalanceall_is_running"] = plugin.mutex.locked()
    res["getroute_method"] = plugin.getroute.__name__
    res["maxhops_threshold"] = plugin.maxhops
    res["msatfactor"] = plugin.msatfactor
    res["erringnodes_threshold"] = plugin.erringnodes
    channels = get_open_channels(plugin)
    health_percent = 0.0
    if len(channels) > 1:
        enough_liquidity = get_enough_liquidity_threshold(channels)
        ideal_ratio = get_ideal_ratio(channels, enough_liquidity)
        res["enough_liquidity_threshold"] = enough_liquidity
        res["ideal_liquidity_ratio"] = f"{ideal_ratio * 100:.2f}%"
        for ch in channels:
            liquidity = liquidity_info(ch, enough_liquidity, ideal_ratio)
            health_percent += health_score(liquidity) * int(ch["total_msat"])
        health_percent /= int(sum(ch["total_msat"] for ch in channels))
    else:
        res["enough_liquidity_threshold"] = Millisatoshi(0)
        res["ideal_liquidity_ratio"] = "0%"
    res["liquidity_health"] = f"{health_percent:.2f}%"
    invoices = plugin.rpc.listinvoices()['invoices']
    rebalances = [i for i in invoices if i.get('status') == 'paid' and i.get('label').startswith("Rebalance")]
    total_fee = Millisatoshi(0)
    total_amount = Millisatoshi(0)
    res["total_successful_rebalances"] = len(rebalances)
    # pyln-client does not support the 'status' argument as yet
    # pays = plugin.rpc.listpays(status="complete")["pays"]
    pays = plugin.rpc.listpays()["pays"]
    pays = [p for p in pays if p.get('status') == 'complete']
    for r in rebalances:
        try:
            pay = next(p for p in pays if p["payment_hash"] == r["payment_hash"])
            total_amount += pay["amount_msat"]
            total_fee += pay["amount_sent_msat"] - pay["amount_msat"]
        except Exception:
            res["total_successful_rebalances"] -= 1
    res["total_rebalanced_amount"] = total_amount
    res["total_rebalance_fee"] = total_fee
    if total_amount > Millisatoshi(0):
        res["average_rebalance_fee_ppm"] = round(total_fee / total_amount * 10**6, 2)
    else:
        res["average_rebalance_fee_ppm"] = 0

    avg_forward_fees = get_avg_forward_fees(plugin, [1, 7, 30])
    res['average_forward_fee_ppm_1d'] = avg_forward_fees[0]
    res['average_forward_fee_ppm_7d'] = avg_forward_fees[1]
    res['average_forward_fee_ppm_30d'] = avg_forward_fees[2]

    return res

# Method for getting all peers of a node.
def get_open_channels(plugin: Plugin):
    channels = []
    for peer in plugin.rpc.listpeers()["peers"]:
        for ch in peer["channels"]:
            if ch["state"] == "CHANNELD_NORMAL" and not ch["private"]:
                channels.append(ch)
    return channels

def rebalanceall_thread(plugin: Plugin):
    if not plugin.mutex.acquire(blocking=False):
        return
    try:
        start_ts = time.time()

        feeadjuster_state = feeadjuster_toggle(plugin, False)

        channels = get_open_channels(plugin)
        plugin.enough_liquidity = get_enough_liquidity_threshold(channels)
        plugin.ideal_ratio = get_ideal_ratio(channels, plugin.enough_liquidity)
        plugin.log(f"Automatic rebalance is running with enough liquidity threshold: {plugin.enough_liquidity}, "
                   f"ideal liquidity ratio: {plugin.ideal_ratio * 100:.2f}%, "
                   f"min rebalancable amount: {plugin.min_amount}, "
                   f"feeratio: {plugin.feeratio}")

        failed_channels = []
        success = 0
        fee_spent = Millisatoshi(0)
        while not plugin.rebalance_stop:
            result = maybe_rebalance_once(plugin, failed_channels)
            if not result["success"]:
                break
            success += 1
            fee_spent += result["fee_spent"]
        feeadjust_would_be_nice(plugin)
        feeadjuster_toggle(plugin, feeadjuster_state)
        elapsed_time = timedelta(seconds=time.time() - start_ts)
        plugin.rebalanceall_msg = f"Automatic rebalance finished: {success} successful rebalance, {fee_spent} fee spent, it took {str(elapsed_time)[:-3]}"
        plugin.log(plugin.rebalanceall_msg)
    finally:
        plugin.mutex.release()



plugin.add_option(
    'This is a first implementation of a rebalancing plugin based on Hide & Seek protocol. '
    'You can find all preliminaries and theoretical background at https://arxiv.org/pdf/2110.08848.pdf')

plugin.run()
