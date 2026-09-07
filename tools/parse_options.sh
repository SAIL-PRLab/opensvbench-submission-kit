#!/bin/bash

# Parse command-line options.
# Example usage:
#   . parse_options.sh || exit 1;
# It parses variables in the main script with the pattern:
#   variable=value

while true; do
  if (( $# == 0 )); then
    break
  fi

  if [[ "$1" == --* ]]; then
    opt="$1"
    shift
    var="${opt/--/}"
    var="${var//-/_}"   # hyphen -> underscore (e.g. max-utts -> max_utts)
    if [[ ! "${var}" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
      echo "Error: invalid option name: ${opt}"
      return 1
    fi
    if (( $# == 0 )) || [[ -z "$1" ]]; then
      echo "Error: missing value after $opt"
      return 1
    fi
    printf -v "${var}" '%s' "$1"
    shift
  else
    break
  fi
done

return 0
