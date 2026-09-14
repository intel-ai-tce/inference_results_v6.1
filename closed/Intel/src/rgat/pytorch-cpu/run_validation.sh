#!/bin/bash

FILENAME=${1}
CHECKSUM=${2}
if [ -d "${FILENAME}" ] ; then
    cd ${FILENAME}
    FOUND=$(find . -type f -exec md5sum {} + | LC_ALL=C sort | md5sum | cut -d' ' -f1)
elif [ -f "${FILENAME}" ]; then
    FOUND=$(find ${FILENAME} -type f -exec md5sum {} + | cut -d' ' -f1)
else
    FOUND="N/A"
fi

if [ "${CHECKSUM}" == "${FOUND}" ]; then
    echo "[MATCH] ${FILENAME}: ${CHECKSUM}"
else
    echo "[ERROR] ${FILENAME}: Expected (${CHECKSUM}) vs Found (${FOUND})"
fi