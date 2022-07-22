#!/usr/bin/python3
import sys, os, time
from lightning.lightning import LightningRpc, RpcError
from pathlib import Path
from datetime import timedelta, datetime

ts_format='%Y-%m-%d %H:%M:%S'

def is_recent(line):
	ts = datetime.strptime(line.split(': ')[0], ts_format)
	rv = datetime.now() - ts < timedelta(seconds=24*60*60)
	return rv

def find_route(target, source, msatoshi, my_node_id):
	logs = ""
	try:
		f = open(os.path.splitext(sys.argv[0])[0] + ".log", "r")
		lines = [line for line in f.readlines() if is_recent(line)]
		logs = " ".join(line.strip() for line in lines)
	except:
		pass
	try:
		for i in range(10):
			route = l.getroute(target, msatoshi, riskfactor=1, cltv=9, fromid=source)
			if any(r['id'] == my_node_id for r in route['route']):
				continue
			if any(r['channel'] in logs for r in route['route']):
				continue
			return route['route']
	except:
		pass
	return False

def setup_routing_fees(route, msatoshi):
	delay = 9
	for r in reversed(route):
		r['msatoshi'] = msatoshi
		r['amount_msat'] = str(msatoshi)+"msat"
		r['delay'] = delay
		channels = l.listchannels(r['channel'])
		for ch in channels.get("channels"):
			if ch['destination'] == r['id']:
				fee = ch['base_fee_millisatoshi']
				fee += msatoshi * ch['fee_per_millionth'] / 1000000
				msatoshi += round(fee)
				delay += ch['delay']

def get_channel_to_peer(peer_id):
	try:
		peer = l.listpeers(peer_id)
		if peer["peers"][0]["channels"][0]["state"] == "CHANNELD_NORMAL":
			return peer["peers"][0]["channels"][0]["short_channel_id"]
	except:
		pass
	print("Cannot find route to peer: " + peer_id)
	exit(1)

if len(sys.argv) < 4:
	print("Usage:\n%s outgoing_node_id incoming_node_id msatoshi"%(sys.argv[0]))
	exit(0)

outgoing_node_id = sys.argv[1]
incoming_node_id = sys.argv[2]
msatoshi = int(sys.argv[3])

try:
	rpc_file = str(Path.home()) + "/.lightning/lightning-rpc"
	l = LightningRpc(rpc_file)
	my_node_id = l.getinfo().get('id')
except:
	print("Cannot connect to lightning RPC Socket at %s"%(rpc_file))
	exit(1)

print("My node id: " + my_node_id)

route_out = {'id': outgoing_node_id}
route_out['channel'] = get_channel_to_peer(outgoing_node_id)
print("Outgoing node: %s, channel: %s"%(route_out['id'], route_out['channel']))

route_in = {'id': my_node_id}
route_in['channel'] = get_channel_to_peer(incoming_node_id)
print("Incoming node: %s, channel: %s"%(incoming_node_id, route_in['channel']))

label = "redistribute"+str(int(time.time()))
invoice = l.invoice(msatoshi, label, "redistribute")
payment_hash = invoice['payment_hash']
print("Invoice payment_hash: %s"%payment_hash)

for i in range(5):
	route_mid = find_route(incoming_node_id, outgoing_node_id, msatoshi, my_node_id)
	if route_mid == False:
		print("Cannot find route for %d. attempt"%(i + 1))
		continue
	route = [route_out] + route_mid + [route_in]
	setup_routing_fees(route, msatoshi)
	print('\nPayment is going on the following route...')
	for r in route:
		print("Node: %s, channel: %13s, %d msat"%(r['id'], r['channel'], r['msatoshi']))
	fees = route[0]['msatoshi'] - route[-1]['msatoshi']
	print("Route contains %d nodes, pays: %d msat with %d msat fee"%
	   (len(route), msatoshi, fees))
	try:
		input("Press Enter to continue or Ctrl-C to skip...")
		l.sendpay(route, payment_hash)
		l.waitsendpay(payment_hash)
		print("Redistribute SUCCEEDED")
		exit(0)
	except RpcError as e:
		logfile = os.path.splitext(sys.argv[0])[0] + ".log"
		print("%d. attempt failed, writing logs to %s"%(i + 1, logfile))
		with open(logfile, 'a') as outfile:
			ts = datetime.now().strftime(ts_format)
			outfile.write("%s: %s\n"%(ts, str(e)))
		if incoming_node_id in str(e):
			print("Error with incoming node, exiting.")
			break
		if outgoing_node_id in str(e):
			print("Error with outgoing node, exiting.")
			break
	except KeyboardInterrupt:
		print("\n%d. attempt skipped"%(i + 1))
print("Redistribute FAILED")
l.delinvoice(label, "unpaid")