# tree_project.sh
#!/usr/bin/env bash

# Description: Show filtered project tree summary
echo "📁 Project structure (filtered):"
echo

tree . \
  -I '__pycache__|.*\.pyc|.*\.log|.*\.db.*|.*\.egg-info|.*\.sqlite|\.git' \
  --dirsfirst \
  -a \
  -L 3

