#!/usr/bin/env bash
## Start script for lightning daemon.

set -E

. $LIBPATH/util/timers.sh

###############################################################################
# Environment
###############################################################################

DATA_PATH="/data/lightning"
PLUG_PATH="$HOME/run/plugins"
PEER_PATH="$SHAREPATH/$HOSTNAME"
LOGS_PATH="/var/log/lightning"

PEER_FILE="$PEER_PATH/lightning-peer.conf"
FUND_FILE="$DATA_PATH/fund.address"
LOGS_FILE="$LOGS_PATH/lightningd.log"

DEFAULT_PEER_PORT=19846
DEFAULT_PEER_TIMEOUT=60
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

###############################################################################
# Script
###############################################################################

if [ -z "$(pgrep bitcoind)" ]; then echo "Bitcoind is not running!" && exit 1; fi

templ banner "Lightning Core Configuration"

## Create any missing paths.
[ ! -d "$DATA_PATH" ] && mkdir -p "$DATA_PATH"
[ ! -d "$LOGS_PATH" ] && mkdir -p "$LOGS_PATH"

## Clear logs.
##if [ -e "$LOGS_FILE" ]; then rm $LOGS_FILE && touch $LOGS_FILE; fi

## Start lightning daemon.
$LIBPATH/start/lightning/lightningd-start.sh

## Start CL-REST Server
[ -n "$REST_NODE" ] && $LIBPATH/start/lightning/cl-rest-start.sh

###############################################################################
# Payment Configuration
###############################################################################

## Generate a funding address.
if [ ! -e "$FUND_FILE" ] || [ -z "$(cat $FUND_FILE)" ]; then
  printf "Generating new payment address for lightning"
  lightning-cli newaddr | jgrep bech32 > $FUND_FILE
  templ ok
fi

## Configure channel settings.
lightning-cli funderupdate match 100 > /dev/null 2>&1
lightning-cli autocleaninvoice 60 > /dev/null 2>&1
lightning-cli funderupdate -k lease_fee_base_msat=0 lease_fee_basis=0 funding_weight=0 > /dev/null 2>&1

###############################################################################
# Plugins
###############################################################################

## Enable plugins
if [ -d "$PLUG_PATH" ]; then
  plugins=`find $PLUG_PATH -maxdepth 1 -type d`
  if [ -n "$plugins" ]; then
    echo && printf "Plugins:\n"
    for plugin in $plugins; do
      name=$(basename $plugin)
      if case $name in .*) ;; *) false;; esac; then continue; fi
      plugin_name=`find $plugin -maxdepth 1 -name $name.*`
      if [ -e "$plugin_name" ]; then
        printf "$IND Enabling $name plugin"
        chmod +x $plugin_name
        lightning-cli plugin start $plugin_name > /dev/null 2>&1
        templ ok
      fi
    done
  fi
fi

###############################################################################
# Peer Connection
###############################################################################

## Set defaults.
[ -z "$CLN_PEER_PORT" ] && CLN_PEER_PORT=$DEFAULT_PEER_PORT
[ -z "$PEER_TIMEOUT" ]  && PEER_TIMEOUT="$DEFAULT_PEER_TIMEOUT"
[ -z "$TOR_TIMEOUT" ]   && TOR_TIMEOUT="$DEFAULT_TOR_TIMEOUT"

[ -n "$(pgrep tor)" ] \
  && CONN_TIMEOUT="$TOR_TIMEOUT" \
  || CONN_TIMEOUT="$PEER_TIMEOUT"

if ( [ -n "$PEER_LIST" ] || [ -n "$CHAN_LIST" ] ); then
  for peer in $(printf $PEER_LIST $CHAN_LIST | tr ',' ' '); do

    [ "$peer" = "$HOSTNAME" ] && continue
    
    ## Search for peer file in peers path.
    echo && printf "Checking connection to $peer:"
    config=`get_peer_config $peer`

    ## Exit out if peer file is not found.
    [ ! -e "$config" ] && templ fail && continue

    ## Parse current peering info.
    onion_host=`cat $config | kgrep ONION_NAME`
    node_id="$(cat $config | kgrep NODE_ID)"
    if [ -z "$LOCAL_ONLY" ] && [ -n "$(pgrep tor)" ] && [ -n "$onion_host" ]; then
      peer_host="$onion_host"
    else
      peer_host="$(cat $config | kgrep HOST_NAME)"
    fi

    ## If valid peer, then connect to node.
    if ! is_node_configured "$node_id"; then
      printf "\n$IND Adding node: $(prevstr $node_id)@$(prevstr -l 20 $peer_host):$CLN_PEER_PORT"
      lightning-cli connect "$node_id@$peer_host:$CLN_PEER_PORT" > /dev/null 2>&1
      printf "\n$IND Connecting to node"
    fi

    ( while ! is_node_connected $node_id; do sleep 1 && printf "."; done; ) & timeout_child $CONN_TIMEOUT
    is_node_connected $node_id && templ conn || templ tout

  done
fi
