import os
from pyln.testing.fixtures import *


# plugin_path = os.path.join(os.path.dirname(__file__), "test.py")
# pluginopt = {'plugin': plugin_path}
currdir = os.path.dirname(__file__)
plugin = os.path.join(currdir, 'test.py')


def test_plugin_starts(node_factory):
    print(plugin)
    #l1 = node_factory.get_node()
    # Test dynamically
    l1, l2, l3, l4, l5 = node_factory.get_nodes(5)
    channels = [(l1, l2), (l3, l2), (l3, l4), (l2, l5), (l5, l3)]
    for src, dst in channels:
        src.openchannel(dst, capacity=10**6)

    assert (channels != 0)

def test_plugin_amount_of_nodes(node_factory):
    print(plugin)
    #l1 = node_factory.get_node()
    # Test dynamically
    l1, l2, l3, l4, l5 = node_factory.get_nodes(5)
    channels = [(l1, l2), (l3, l2), (l3, l4), (l2, l5), (l5, l3)]
    for src, dst in channels:
        src.openchannel(dst, capacity=10**6)

    assert (len(channels) == 6)


def capital_case(x):
    return x.capitalize()


def test_capital_case():
    assert capital_case('semaphore') == 'Semaphore'

    #l1.rpc.plugin_start(plugin)
    #l1.rpc.plugin_stop(plugin)
    #l1.rpc.plugin_start(plugin)
    #l1.stop()
    # Then statically
    #l1.daemon.opts["plugin"] = plugin_path
    #l1.start()


# def test_your_plugin(node_factory, bitcoind):
#     l1 = node_factory.get_node(options=pluginopt)
#     s = l1.rpc.getinfo()
#     assert(s['network'] == 'regtest')


# def test_no_possible_rebalancing(node_factory):
#     opts = [{}, {'plugin': plugin_path}, {}, {}, {}]
#     l1, l2, l3, l4, l5 = node_factory.get_nodes(5, opts=opts)
