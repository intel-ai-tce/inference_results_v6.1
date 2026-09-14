#!/bin/bash
set -e

HEAD_IP=$(hostname -I | awk '{print $1}')
echo "HEAD NODE IP ADDRESS $HEAD_IP"
ray start --disable-usage-stats --head --port=6379
