#!/bin/bash
# Script 1: Setup Kubectl Aliases and Autocompletion
# Adds common kubectl aliases and enables shell autocompletion.

# --- Detect Shell and Set Config File ---
CURRENT_SHELL=$(basename "$SHELL")
CONFIG_FILE=""
SHELL_TYPE=""

echo "INFO (Script 1): Detecting your shell..."

if [ "$CURRENT_SHELL" = "bash" ]; then
  CONFIG_FILE="$HOME/.bashrc"
  SHELL_TYPE="bash"
  echo "INFO (Script 1): Detected Bash. Configuration file: $CONFIG_FILE"
elif [ "$CURRENT_SHELL" = "zsh" ]; then
  CONFIG_FILE="$HOME/.zshrc"
  SHELL_TYPE="zsh"
  echo "INFO (Script 1): Detected Zsh. Configuration file: $CONFIG_FILE"
else
  echo "ERROR (Script 1): Unsupported shell '$CURRENT_SHELL'. Only bash and zsh are supported."
  exit 1
fi

# --- Add Header to Config File (if not already present) ---
HEADER_TEXT="# --- Kubernetes Enhancements (added by script) ---"
if ! grep -qF "$HEADER_TEXT" "$CONFIG_FILE" 2>/dev/null; then
  echo "" >> "$CONFIG_FILE" # Add a newline for separation
  echo "$HEADER_TEXT" >> "$CONFIG_FILE"
fi

# --- Add Aliases ---
echo "INFO (Script 1): Adding kubectl aliases..."
# Use a marker to avoid adding aliases multiple times if script is rerun
ALIAS_MARKER="# START: Kubectl Aliases"
if ! grep -qF "$ALIAS_MARKER" "$CONFIG_FILE" 2>/dev/null; then
  echo "$ALIAS_MARKER" >> "$CONFIG_FILE"
  cat << EOF >> "$CONFIG_FILE"

alias k='kubectl'

alias kpvc='kubectl get pvc'
alias kgp='kubectl get pods'
alias kgj='kubectl get jobs'


alias kl='kubectl logs'
alias kf='kubectl logs -f' # Follow logs
alias kd='kubectl describe'
alias ke='kubectl exec -it'

EOF
else
    echo "INFO (Script 1): Kubectl aliases seem to be already added (marker found)."
fi


# --- Add Kubectl Autocompletion ---
COMPLETION_LINE="source <(kubectl completion $SHELL_TYPE)"
# Check if completion is already sourced to avoid duplicates
if ! grep -qF "$COMPLETION_LINE" "$CONFIG_FILE" 2>/dev/null; then
  echo "INFO (Script 1): Adding kubectl autocompletion..."
  echo "$COMPLETION_LINE" >> "$CONFIG_FILE"
else
  echo "INFO (Script 1): Kubectl autocompletion already set up."
fi

# --- Final Instructions ---
echo ""
echo "---------------------------------------------------------------------"
echo "✅ Script 1 (Aliases/Completion) finished!"
echo "Changes appended to: $CONFIG_FILE"
echo ""
echo "IMPORTANT: To apply these changes in your *current* shell session,"
echo "please run the following command:"
echo ""
echo "    source $CONFIG_FILE"
echo ""
echo "New shell sessions will automatically pick up the changes."
echo "---------------------------------------------------------------------"
echo ""
# --- End Script 1 ---