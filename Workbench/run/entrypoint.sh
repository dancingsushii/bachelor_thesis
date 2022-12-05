#!/usr/bin/env bash
## Entrypoint script for image.

set -E

###############################################################################
# Environment
###############################################################################

IND=`fgc 215 "|-"`

###############################################################################
# Methods
###############################################################################

clean_exit() {
  ## If exit code is non-zero, print fail message and clean up.
  status="$?"
  [ $status -ne 0 ] && ( printf "\nFailed with exit code $state"; templ fail )
  [ $status -ne 0 ] && [ -z "$DEVMODE" ] && cleanup || exit 0
}

cleanup() {
  ## Delete share info before exiting.
  if [ -z "$DEVMODE" ]; then
    printf "Delisting $SHAREPATH/$HOSTNAME ... "
    rm -rf "$SHAREPATH/$HOSTNAME"
    printf "done.\n" && exit 2
  fi
}

###############################################################################
# Script
###############################################################################

trap clean_exit EXIT; trap cleanup SIGTERM SIGKILL

[ -z "$DEVMODE" ] && sleep 1   ## Add some delay for docker to attach tty properly.

## Make sure we are in root.
cd /root

## Execute startup scripts.
for script in `find $RUNPATH/startup -name *.sh | sort`; do
  IND=$IND $script; state="$?"
  if [ $state -ne 0 ]; then exit $state; fi
done

## Print a failure banner if we fail.
[ $? -ne 0 ] && ( templ banner "Node startup failed!"; exit 1 )

## Start terminal service.
terminal
