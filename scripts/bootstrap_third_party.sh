#!/usr/bin/env bash
set -euo pipefail

mkdir -p third_party

if [ ! -d third_party/lerobot/.git ]; then
  git clone https://github.com/huggingface/lerobot.git third_party/lerobot
fi

if [ ! -d third_party/LIBERO/.git ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
fi

echo "Third-party checkouts are ready under third_party/."

