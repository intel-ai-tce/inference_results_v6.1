#!/bin/bash
# Run this after env.sh
mkdir -p ${LAB_CLOG}

env | sort >> ${LAB_CLOG}/host-env.txt
