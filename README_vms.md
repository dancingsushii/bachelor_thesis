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
