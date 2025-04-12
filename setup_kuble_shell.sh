#!/bin/bash
# Script: Setup Kubectl Aliases and Autocompletion
# Appends a block of aliases and completion setup if a header marker isn't found.

# --- Configuration ---
K8S_HEADER="# --- Kubernetes Enhancements (added by script) ---"

# --- Detect Shell ---
CURRENT_SHELL=$(basename "$SHELL")
CONFIG_FILE=""
SHELL_TYPE=""

echo "INFO: Detecting your shell..."

# Determine shell and config file
case "$CURRENT_SHELL" in
  bash)
    CONFIG_FILE="$HOME/.bashrc"
    SHELL_TYPE="bash"
    echo "INFO: Detected Bash. Configuration file: $CONFIG_FILE"
    ;;
  zsh)
    CONFIG_FILE="$HOME/.zshrc"
    SHELL_TYPE="zsh"
    echo "INFO: Detected Zsh. Configuration file: $CONFIG_FILE"
    ;;
  *)
    echo "ERROR: Unsupported shell '$CURRENT_SHELL'. Only bash and zsh are supported." >&2
    exit 1
    ;;
esac

# Check if config file exists, create if not
if [ ! -f "$CONFIG_FILE" ]; then
    echo "WARNING: Configuration file '$CONFIG_FILE' not found. Creating it."
    touch "$CONFIG_FILE" || { echo "ERROR: Could not create $CONFIG_FILE" >&2; exit 1; }
fi

# --- Check if Enhancements Already Added ---
# Simple check using the header text as a marker
echo "INFO: Checking if Kubernetes enhancements seem already added..."
if grep -qF "$K8S_HEADER" "$CONFIG_FILE" 2>/dev/null; then
  echo "INFO: Marker '$K8S_HEADER' found in $CONFIG_FILE."
  echo "INFO: Assuming setup is already present. Skipping additions."
  changes_made=0
else
  # --- Add Enhancements Block ---
  echo "INFO: Marker not found. Adding Kubernetes enhancements block to $CONFIG_FILE..."

  # Use a Heredoc (cat << EOF) to append the whole block.
  # $SHELL_TYPE will be correctly substituted in the completion line.
  cat << EOF >> "$CONFIG_FILE"

$K8S_HEADER
alias k='kubectl'
alias kgp='kubectl get pods'
alias kgd='kubectl get deployment'
alias kgs='kubectl get service'
alias kpvc='kubectl get pvc'
alias kgj='kubectl get jobs'
alias ka='kubectl apply -f'
alias kl='kubectl logs'
alias kf='kubectl logs -f' # Follow logs
alias kd='kubectl describe'
alias ke='kubectl exec -it'

# Autocompletion Setup (only sources if kubectl command exists)
if command -v kubectl &> /dev/null; then
  source <(kubectl completion $SHELL_TYPE)
fi
# --- End Kubernetes Enhancements ---

EOF
  echo "INFO: Block added successfully."
  changes_made=1 # Flag that we made changes
fi

# --- Final Instructions ---
echo ""
echo "---------------------------------------------------------------------"
if [ "$changes_made" -eq 1 ]; then
    echo "✅ Script finished! Enhancements added to: $CONFIG_FILE"
    echo ""
    echo "IMPORTANT: To apply these changes in your *current* shell session,"
    echo "please run the following command:"
    echo ""
    echo "    source \"$CONFIG_FILE\""
    echo ""
    echo "New shell sessions will automatically pick up the changes."
else
    echo "✅ Script finished! No changes made as the marker was found in: $CONFIG_FILE"
fi
echo "---------------------------------------------------------------------"
echo ""

# --- End Script ---