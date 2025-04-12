#!/bin/bash
# Script 2: Setup Shell Auto-Reloading
# Adds logic to automatically source the shell config file when it changes.

# --- Detect Shell and Set Config File ---
CURRENT_SHELL=$(basename "$SHELL")
CONFIG_FILE=""
SHELL_TYPE=""

echo "INFO (Script 2): Detecting your shell..."

if [ "$CURRENT_SHELL" = "bash" ]; then
  CONFIG_FILE="$HOME/.bashrc"
  SHELL_TYPE="bash"
  echo "INFO (Script 2): Detected Bash. Configuration file: $CONFIG_FILE"
elif [ "$CURRENT_SHELL" = "zsh" ]; then
  CONFIG_FILE="$HOME/.zshrc"
  SHELL_TYPE="zsh"
  echo "INFO (Script 2): Detected Zsh. Configuration file: $CONFIG_FILE"
else
  echo "ERROR (Script 2): Unsupported shell '$CURRENT_SHELL'. Only bash and zsh are supported."
  exit 1
fi

# --- Add Automatic Re-sourcing Logic ---
# Use unique markers to avoid adding the logic multiple times
AUTO_RELOAD_MARKER_START="# START: Auto-reload $SHELL_TYPE config (added by script)"
AUTO_RELOAD_MARKER_END="# END: Auto-reload $SHELL_TYPE config"

if ! grep -qF "$AUTO_RELOAD_MARKER_START" "$CONFIG_FILE" 2>/dev/null; then
  echo "INFO (Script 2): Adding automatic config reload logic..."
  echo "" >> "$CONFIG_FILE" # Add a newline for separation
  echo "$AUTO_RELOAD_MARKER_START" >> "$CONFIG_FILE"

  if [ "$SHELL_TYPE" = "bash" ]; then
    # Bash auto-reload logic (handles GNU and BSD stat)
    cat << 'EOF' >> "$CONFIG_FILE"

_bashrc_last_mod_time=0
_update_bashrc_if_needed() {
  local config_file="$HOME/.bashrc"
  local current_mod_time=0
  if [[ ! -f "$config_file" ]]; then return; fi # Skip if file doesn't exist
  # Check which stat command is available
  if stat -c %Y "$config_file" > /dev/null 2>&1; then # GNU stat
    current_mod_time=$(stat -c %Y "$config_file")
  elif stat -f %m "$config_file" > /dev/null 2>&1; then # BSD stat (macOS)
    current_mod_time=$(stat -f %m "$config_file")
  else
    echo "WARN: Cannot determine modification time for $config_file. Auto-reload disabled." >&2
    PROMPT_COMMAND=${PROMPT_COMMAND//_update_bashrc_if_needed;} # Attempt removal
    return
  fi

  if [[ "$_bashrc_last_mod_time" -eq 0 ]]; then _bashrc_last_mod_time=$current_mod_time; fi

  if [[ "$current_mod_time" -gt "$_bashrc_last_mod_time" ]]; then
    echo "[bashrc] Detected change, reloading..." >&2
    source "$config_file"
    _bashrc_last_mod_time=$current_mod_time
  fi
}
# Append the check to PROMPT_COMMAND, preserving existing commands
if [[ -z "$PROMPT_COMMAND" || "$PROMPT_COMMAND" != *"_update_bashrc_if_needed"* ]]; then
    PROMPT_COMMAND="_update_bashrc_if_needed${PROMPT_COMMAND:+; $PROMPT_COMMAND}"
fi
# Initialize mod time
_update_bashrc_if_needed > /dev/null

EOF

  elif [ "$SHELL_TYPE" = "zsh" ]; then
    # Zsh auto-reload logic (handles GNU and BSD stat)
    cat << 'EOF' >> "$CONFIG_FILE"

_zshrc_last_mod_time=0
_check_zshrc_update() {
  local config_file="$HOME/.zshrc"
  local current_mod_time=0
  if [[ ! -f "$config_file" ]]; then return; fi # Skip if file doesn't exist
  # Check which stat command is available
  if stat -c %Y "$config_file" > /dev/null 2>&1; then # GNU stat
    current_mod_time=$(stat -c %Y "$config_file")
  elif stat -f %m "$config_file" > /dev/null 2>&1; then # BSD stat (macOS)
    current_mod_time=$(stat -f %m "$config_file")
  else
     echo "WARN: Cannot determine modification time for $config_file. Auto-reload disabled." >&2
     return
  fi

  if [[ "$_zshrc_last_mod_time" -eq 0 ]]; then _zshrc_last_mod_time=$current_mod_time; fi

  if [[ "$current_mod_time" -gt "$_zshrc_last_mod_time" ]]; then
    echo "[zshrc] Detected change, reloading..." >&2
    source "$config_file"
    _zshrc_last_mod_time=$current_mod_time
  fi
}

# Load add-zsh-hook if necessary and available
if command -v autoload >/dev/null && ! (( ${+functions[add-zsh-hook]} )); then
    autoload -Uz add-zsh-hook
fi

# Check if hook already exists before adding (requires add-zsh-hook loaded)
if (( ${+functions[add-zsh-hook]} )); then
    local _hook_exists=0
    if (( ${+precmd_functions} )); then
      for _hook in "${precmd_functions[@]}"; do
        if [[ "$_hook" == "_check_zshrc_update" ]]; then
          _hook_exists=1
          break
        fi
      done
    fi

    if [[ $_hook_exists -eq 0 ]]; then
      add-zsh-hook precmd _check_zshrc_update
    fi
    # Clean up temporary variable if it was declared
    unset _hook_exists
else
    # Fallback or warning if add-zsh-hook is unavailable
    echo "WARN: zsh add-zsh-hook mechanism not found. Auto-reload might not be added correctly." >&2
fi
# Initialize mod time
_check_zshrc_update > /dev/null

EOF
  fi # End if SHELL_TYPE

  echo "$AUTO_RELOAD_MARKER_END" >> "$CONFIG_FILE"
else
    echo "INFO (Script 2): Automatic config reload logic already present."
fi # End check for AUTO_RELOAD_MARKER_START


# --- Final Instructions ---
echo ""
echo "---------------------------------------------------------------------"
echo "✅ Script 2 (Auto-Reload) finished!"
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
# --- End Script 2 ---