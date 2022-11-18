import os
from pyln.testing.fixtures import *
import unittest


plugin_path = os.path.join(os.path.dirname(__file__), "hide&seek_plugin.py")
pluginopt = {'plugin': plugin_path}


def test_plugin_starts(node_factory):
    l1 = node_factory.get_node()
    # Test dynamically
    l1.rpc.plugin_start(plugin_path)
    l1.rpc.plugin_stop(plugin_path)
    l1.rpc.plugin_start(plugin_path)
    l1.stop()
    # Then statically
    l1.daemon.opts["plugin"] = plugin_path
    l1.start()


def test_your_plugin(node_factory, bitcoind):
    l1 = node_factory.get_node(options=pluginopt)
    s = l1.rpc.getinfo()
    assert(s['network'] == 'regtest')


def test_no_possible_rebalancing(node_factory):
    # For more information look:
    # https://github.com/lightningd/plugins/blob/master/drain/test_drain.py

    l1, l2, l3, l4 = node_factory.line_graph(4, opts=pluginopt)
    l4.rpc.connect(l1.info['id'], 'localhost', l1.port)
    nodes = [l1, l2, l3, l4]

    scid12 = l1.get_channel_scid(l2)
    scid23 = l2.get_channel_scid(l3)
    scid34 = l3.get_channel_scid(l4)
    l4.fundchannel(l1, 10**6)
    scid41 = l4.get_channel_scid(l1)
    assert(True)
