#!/bin/bash

SSHDIR=$HOME/ssh_in_docker_home

mkdir -p $SSHDIR

chmod go-w $HOME
chmod 700 $HOME/.ssh
chmod 600 $HOME/.ssh/authorized_keys

rm -f $SSHDIR/ssh_host_*
ssh-keygen -q -N "" -t dsa -f $SSHDIR/ssh_host_dsa_key
ssh-keygen -q -N "" -t rsa -b 4096 -f $SSHDIR/ssh_host_rsa_key
ssh-keygen -q -N "" -t ecdsa -f $SSHDIR/ssh_host_ecdsa_key
ssh-keygen -q -N "" -t ed25519 -f $SSHDIR/ssh_host_ed25519_key

cat >$SSHDIR/sshd_config <<EOF
## Use a non-privileged port
Port 6322
## provide the new path containing these host keys
HostKey $SSHDIR/ssh_host_rsa_key
HostKey $SSHDIR/ssh_host_ecdsa_key
HostKey $SSHDIR/ssh_host_ed25519_key
## Enable DEBUG log. You can ignore this but this may help you debug any issue while enabling SSHD for the first time
LogLevel DEBUG3
ChallengeResponseAuthentication no
UsePAM no
X11Forwarding yes
PrintMotd no
PermitTTY yes
AllowTcpForwarding yes
## Provide a path to store PID file which is accessible by normal user for write purpose
PidFile $SSHDIR/sshd.pid
AcceptEnv LANG LC_*
Subsystem sftp internal-sftp
AuthorizedKeysFile	.ssh/authorized_keys
ForceCommand $SSHDIR/singularity-login-wrapper.sh
EOF

cat > $SSHDIR/singularity-login-wrapper.sh <<'EOF'
#!/usr/bin/env bash
set -o pipefail

# 1) Replay the env stage that Singularity's action scripts perform
if [ -d /.singularity.d/env ]; then
  # predictable order
  for s in /.singularity.d/env/*.sh; do
    [ -r "$s" ] && . "$s"
  done
fi

# 2) If a remote command/subsystem was requested, honour it.
#    Keep sftp/scp working; otherwise run the command inside a login shell.
if [ -n "${SSH_ORIGINAL_COMMAND-}" ]; then
  case "$SSH_ORIGINAL_COMMAND" in
    internal-sftp|*sftp-server* )
      exec $SSH_ORIGINAL_COMMAND
      ;;
    scp* )
      exec $SSH_ORIGINAL_COMMAND
      ;;
    * )
      exec "${SHELL:-/bin/bash}" -lc "$SSH_ORIGINAL_COMMAND"
      ;;
  esac
else
  # interactive login
  exec "${SHELL:-/bin/bash}" -l
fi
EOF

#cat >$HOME/.profile <<EOF
#export LD_LIBRARY_PATH=/.singularity.d/libs
#EOF

chmod 600 $SSHDIR/*
chmod 644 $SSHDIR/sshd_config
chown -R $USER $SSHDIR

chmod +x $SSHDIR/singularity-login-wrapper.sh

# Get sshd abs path using where command
SSHD_PATH=/usr/sbin/sshd


while getopts ":b:" opt; do
  case $opt in
    :)
      $SSHD_PATH -D -f $SSHDIR/sshd_config -E $SSHDIR/sshd.log
      exit 0
      ;;
  esac
done

$SSHD_PATH -f $SSHDIR/sshd_config -E $SSHDIR/sshd.log
