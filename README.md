# Welcome to my bachelor repository!

This repository contains all necessary information regarding my bachelor thesis "Vulnerability Assessment and Enhancement of  Current Rebalancing Approaches in Lightning  Network". 

# Virtual machines access

Access on virtual machines works only through `tst1.inet.tu-berlin.de`. From there I can access virtual machines using my private key.

    ssh -i /Users/tetianayakovenko/.ssh/id_ed25519 tetiana@tst1.inet.tu-berlin.de
   
Accessing virtual machines

    ssh -i /afs/net.t-labs.tu-berlin.de/home/tetiana/id_ed25519 root@blockchain1.inet.tu-berlin.de
    ssh -i /afs/net.t-labs.tu-berlin.de/home/tetiana/id_ed25519 root@blockchain2.inet.tu-berlin.de
    ssh -i /afs/net.t-labs.tu-berlin.de/home/tetiana/id_ed25519 root@blockchain3.inet.tu-berlin.de


# Installation on virtual machines

Apart from my personal Raspberry Pi 4 RAM 8 GB running Lightning node not only on testnet but also on mainnet, I got three virtual machines with 8 GB RAM and 600 GB storage each. In order to tun full bitcoin and lightning node on them I did following installations. I first tried install everything with Snap but then I read some negative comments regarding security so I followed classical binary approach.

First of all you need to install sudo and wget to be able to get a binary.

    apt-get update
    apt install sudo
    apt-get install wget

Then I downloaded 64 bit version for linux and extracted a tar ball. Lastly, I installed the binaries in /usr/local/bin directory.

    wget https://bitcoincore.org/bin/bitcoin-core-23.0/bitcoin-23.0-x86_64-linux-gnu.tar.gz
    tar xvzf bitcoin-23.0-x86_64-linux-gnu.tar.gz
    sudo install -m 0755 -o root -g root -t /usr/local/bin bitcoin-23.0/bin/*

Because of the fact that 600 GB (on the day of installation 21.07.22 [blockchain weighed 417.42 GB](https://ycharts.com/indicators/bitcoin_blockchain_size)) was still not enough and I got an error "Disk Space is too low!" I was forced to enable pruning. For this purpose I modified configuration file `bitcoin.conf` and added `prune=550`.  Initially there was no such file, so I had to create it first in `$HOME/.bitcoin/`. 

To accept RPC-JSON commands run 

    bitcoind -server
 
 For the first time this command will start downloading the blockchain so it can last for some time. In my case it was 8 hours because of the prunning.

# Installation on Raspberry Pi 4

To be done.

# Useful links

During research for my topic I found a plenty of useful ressources, libraries, papers, enterpreneurs, podcasts etc. I collect them and listed there for better usage.

# Thesis link

https://www.overleaf.com/project/60a116036e229ad6f3caaf23

# Technical details

## Useful commands
The bunch of problems I've faced during mz bachelor thesis creation.
1. I had some problems with dowloading a testnet chain. I have constanly received an error "Peer=N is stalling block download, disconnecting". I had some concerns this could be because of some hardware problems or DNS settings bit then I found out that mainnet is downloading smoothly. So I read some advices [here](https://github.com/bitcoin/bitcoin/issues/11037), [here](https://bitcointalk.org/index.php?topic=1666725.0) and [here](https://github.com/bitcoin/bitcoin/issues/8518). And I have some thoughts about it:


## Always relevant questions

 1. [Why should I use a particular user and not root for bitcoin running?](https://bitcoin.stackexchange.com/questions/46562/newbie-question-bitcoind-installation-doubte)
 2. 

## Useful commands
Report file system disk space usage. The `-h` flag provides human readable output (which makes reading of the output - easier).

```
df -h
```
Listing open ports in use.

```
sudo ss -tulpn | grep LISTEN
```

## Useful links (to be sorted)
https://vhernando.github.io/run-bitcoin-node-debian-how-to

https://bitcointalk.org/index.php?topic=5395925.0

https://jlopp.github.io/bitcoin-core-config-generator/#config=eyJjaGFpbiI6eyJjaGFpbiI6InRlc3QifX0=

https://medium.com/@retprogramisto/how-to-install-a-pruned-bitcoin-full-node-on-ubuntu-vps-2b81fe170ddf
