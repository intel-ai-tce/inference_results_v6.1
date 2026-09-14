#!/bin/bash
set -e

if [ -n "$1" ]; then  
    HEAD_IP="$1"  
fi 

if [ -z "$HEAD_IP" ]; then  
    echo "Error: No head_ip is provided. Please provide the head_ip as an argument or set the HEAD_IP environment variable."  
    echo "Usage: ./script.sh <head_ip>  OR  HEAD_IP=value ./script.sh"  
    exit 1  
fi  

ray start --disable-usage-stats --address="$HEAD_IP:6379"
