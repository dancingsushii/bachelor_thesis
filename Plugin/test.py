#!/usr/bin/env bash
## Graph connection setup script

set -E

. $LIBPATH/util/timers.sh

PEER_PATH="$SHAREPATH/$HOSTNAME"
PEER_FILE="$PEER_PATH/lightning-peer.conf"
GRAPH_STRUCTURE_WRITE_SCRIPT="$HOME/run/graphlib/setup_graph.py"
GRAPH_STRUCTURE_FILE="$SHAREPATH/graph_pickle.gpickle.gz"
GRAPH_CONFIG_DIR="$HOME/run/graphlib/configs"

DEFAULT_PEER_PORT=19846
DEFAULT_PEER_TIMEOUT=2
DEFAULT_TOR_TIMEOUT=60

###############################################################################
# Methods
###############################################################################

get_peer_config() {
  [ -n "$1" ] && find "$SHAREPATH/$1"* -name lightning-peer.conf 2>&1
}

is_node_configured() {
  [ -n "$1" ] && [ -n "$(lnpy getpeerlist | grep $1)" ]
}

is_node_connected() {
  [ -n "$1" ] && [ "$(lnpy is_peer_connected $1)" -eq 1 ]
}

is_channel_confirmed() {
  [ -n "$1" ] && [ "$(lnpy peerchannelcount "$1")" != "0" ]
}

is_channel_funded() {
  [ -n "$1" ] && [ "$(lnpy peerchannelbalance "$1")" != "0" ]
}

# Block until the given file appears or the given timeout is reached.
# Exit status is 0 iff the file exists.
wait_file() {
  local file="$1"; shift
  local wait_seconds="${1:-10}"; shift # 10 seconds as default timeout
  test $wait_seconds -lt 1 && echo 'At least 1 second is required' && return 1

  until test $((wait_seconds--)) -eq 0 -o -e "$file" ; do sleep 1; done

  test $wait_seconds -ge 0 # equivalent: let ++wait_seconds
}

## Exit out if pickled graph file not found
own_alias=`cat /etc/hostname`
if [ "$own_alias" = 'master.regtest.node' ]; then exit 0; fi
if [ ! -e "$GRAPH_STRUCTURE_FILE" ]; then exit 0; fi

## Indicate readiness
readiness_file="/share/${own_alias}_readiness"
echo "${own_alias} is ready to accept connections!" >> "$readiness_file"
echo && printf "Indicated my readiness by creating $readiness_file"

## Set defaults.
[ -z "$CLN_PEER_PORT" ] && CLN_PEER_PORT=$DEFAULT_PEER_PORT
[ -z "$PEER_TIMEOUT" ]  && PEER_TIMEOUT="$DEFAULT_PEER_TIMEOUT"
[ -z "$TOR_TIMEOUT" ]   && TOR_TIMEOUT="$DEFAULT_TOR_TIMEOUT"

[ -n "$(pgrep tor)" ] \
  && CONN_TIMEOUT="$TOR_TIMEOUT" \
  || CONN_TIMEOUT="$PEER_TIMEOUT"

templ banner "Setting the graph structure according to saved networkx"

echo && printf "Executing the python script to prepare the graph structure setup\n"

python3 "$GRAPH_STRUCTURE_WRITE_SCRIPT"

peer_configs=`find $GRAPH_CONFIG_DIR -maxdepth 1 -type f -name *.conf`
if [ -n "$peer_configs" ]; then
    echo && printf "Configuring peers:\n"
    for peer_config in $peer_configs; do
        echo && printf "Looking at $peer_config"
        # get graph config properties
        peer_alias=`cat $peer_config | kgrep PEER_ALIAS`
        our_funding=`cat $peer_config | kgrep OUR_FUNDING`
        their_funding=`cat $peer_config | kgrep THEIR_FUNDING`
        peer_lightning_config_file=`get_peer_config $peer_alias`
        if [ ! -e "$peer_lightning_config_file" ]; then
            ## Await the creation of the <alias>.regtest.node folders and corresponding lightning-peer.conf files if neccessary
            echo && printf "$peer_alias lightning configuration file did not exist in share when requested, waiting for its creation...\n"
            peer_lightning_config_file="$SHAREPATH/$peer_alias.regtest.node/lightning-peer.conf"
            wait_file "$peer_lightning_config_file" 30 || {
                echo "ERROR: $peer_lightning_config_file missing after waiting 30 seconds\n"
                templ fail
                exit 1
            }
        fi

        ## Check readiness file existence
        peers_readiness_file="/share/${peer_alias}.regtest.node_readiness"
        echo && printf "Waiting for $peers_readiness_file readiness file to appear!\n"
        wait_file "$peers_readiness_file" 60 || {
            echo "ERROR: ${peer_alias} readiness not indicated after one minute of waiting!\n"
            templ fail
            exit 1
        }
        echo && printf "$peers_readiness_file readiness file appeared, connecting to peer!\n"

        ###############################################################################
        # Peer Connection
        ###############################################################################

        ## Parse current peering info.
        onion_host=`cat $peer_lightning_config_file | kgrep ONION_NAME`
        node_id="$(cat $peer_lightning_config_file | kgrep NODE_ID)"
        if [ -z "$LOCAL_ONLY" ] && [ -n "$(pgrep tor)" ] && [ -n "$onion_host" ]; then
            peer_host="$onion_host"
        else
            peer_host="$(cat $peer_lightning_config_file | kgrep HOST_NAME)"
        fi

        ## If valid peer, then connect to node.
        if ! is_node_configured "$node_id"; then
            printf "\n$IND Adding node: $(prevstr $node_id)@$(prevstr -l 20 $peer_host):$CLN_PEER_PORT"
            lightning-cli connect "$node_id@$peer_host:$CLN_PEER_PORT" > /dev/null 2>&1
            printf "\n$IND Connecting to node"
            # wait for channel to get established
            while ! is_node_connected $node_id > /dev/null 2>&1; do sleep 1.5 && printf "."; done; 
        fi

        ( while ! is_node_connected $node_id; do sleep 1 && printf "."; done; ) & timeout_child $CONN_TIMEOUT
        is_node_connected $node_id && templ conn || templ tout

        ###############################################################################
        # Channel Funding
        ###############################################################################

        ## If valid peer, then connect to node.
        if is_node_connected $node_id; then
            if ! is_channel_confirmed $node_id; then
                printf "$IND Opening channel with $peer_alias for $our_funding sats capacity and push $their_funding sats to them.\n"
                printf "$IND Waiting for channel to confirm .\n"
                lightning-cli fundchannel id=$node_id amount="${our_funding}sat" push_msat="${their_funding}sat"> /dev/null 2>&1
                while ! is_channel_funded $node_id > /dev/null 2>&1; do sleep 1.5 && printf "."; done; templ ok
            fi
            printf "$IND Channel balance:"; templ brkt "$(lnpy peerchannelbalance $node_id)"
        else
            printf "$IND No connection to $peer!" && templ fail
        fi

    done
fi

## Indicate that your channels are setup according to graphspawner.conf
node_channel_setup_file="/share/${own_alias}_channels_setup"
echo "${own_alias} has setup all the provided channels!" >> "$node_channel_setup_file"
echo && printf "Indicated that my channels are setup by creating $node_channel_setup_file"
