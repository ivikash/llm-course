#!/usr/bin/env bash
# Regenerate site/files.json with all .md files in the course.
set -euo pipefail
cd "$(dirname "$0")/.."
{
  echo '['
  find . -maxdepth 2 -name '*.md' -not -path './site/*' \
    | sed 's|^\./||' | sort \
    | awk 'BEGIN{first=1} {if(!first)printf ",\n"; first=0; printf "  \"%s\"", $0} END{print ""}'
  echo ']'
} > site/files.json
echo "wrote site/files.json ($(grep -c '"' site/files.json | awk '{print $1/2}') files)"
