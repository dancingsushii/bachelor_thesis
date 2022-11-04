# Welcome to my bachelor repository!

This repository contains all necessary information regarding my bachelor thesis "Vulnerability Assessment and Enhancement of  Current Rebalancing Approaches in Lightning  Network". 


# Raspberry Pi 4

To be done.

## Isntallation on Raspberry Pi 4

## Access Raspberry full node over SSH and Tor

```
torsocks ssh admin@ugssvwei5ujvp64i733jsubcavzh4gtad5mjb26rws3df2lnwgpwaxqd.onion
```

# Useful links

During research for my topic I found a plenty of useful ressources, libraries, papers, enterpreneurs, podcasts etc. I collect them and listed there for better usage.

# Thesis link

https://www.overleaf.com/project/60a116036e229ad6f3caaf23

# Technical details

## Useful commands
Report file system disk space usage. The `-h` flag provides human readable output (which makes reading of the output - easier).

```
df -h
```


## Assumptions for experiments
1. Free rebalancing (without fee)
2. Every node in the ``regtest` is willing to participate
3. MPC delegates are known for all other nodes (bcs we need their ip addresses in order to execute MPyC LP calculation)

Listing open ports in use.

```
sudo ss -tulpn | grep LISTEN
```
Restart Ubuntu.

```
sudo shutdown -r now
```

## Common mistakes / lessons learned
The bunch of problems I've faced during my bachelor thesis creation.
1. I had some problems with dowloading a testnet chain. I have constanly received an error "Peer=N is stalling block download, disconnecting". I had some concerns this could be because of some hardware problems or DNS settings bit then I found out that mainnet is downloading smoothly. So I read some advices [here](https://github.com/bitcoin/bitcoin/issues/11037), [here](https://bitcointalk.org/index.php?topic=1666725.0) and [here](https://github.com/bitcoin/bitcoin/issues/8518). Which steps I tried: a) maxconnection=1 didn't help, b) -reindex didn't help.


## Always relevant questions

 1. [Why should I use a particular user and not root for bitcoin running?](https://bitcoin.stackexchange.com/questions/46562/newbie-question-bitcoind-installation-doubte)
 2. 


## Useful links (to be sorted)
https://vhernando.github.io/run-bitcoin-node-debian-how-to

https://bitcointalk.org/index.php?topic=5395925.0

https://jlopp.github.io/bitcoin-core-config-generator/#config=eyJjaGFpbiI6eyJjaGFpbiI6InRlc3QifX0=

https://medium.com/@retprogramisto/how-to-install-a-pruned-bitcoin-full-node-on-ubuntu-vps-2b81fe170ddf

## Found rebalancing LN plugins (only for C-Lightning)
1. Main [rebalance plugin](https://github.com/lightningd/plugins/tree/master/rebalance) curated by CLN community. Circular rebalance method, local
2. [Plugin](https://github.com/giovannizotta/circular). Circular rebalance, local.

