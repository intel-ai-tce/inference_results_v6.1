#!/bin/bash

CODE_DIR=$(dirname -- $0)
SCRIPTS_DIR=${CODE_DIR}/scripts
RUN_HEALTHCHECK=${RUN_HEALTHCHECK:-0}
source "$SCRIPTS_DIR/default_log_folder.sh"
source "$SCRIPTS_DIR/utils/health_check.sh"

input_string=$@

config_path=$(echo "$input_string" | grep -oP '(?<=--config-path\s)[^ ]+')
model_name=$(basename "$config_path")

config_name=$(echo "$input_string" | grep -oP '(?<=--config-name\s)[^\s]+')
IFS='_' read -r scenario gpu <<< "$config_name"

output_log_dir=$(echo "$input_string" | grep -oP '(?<=harness_config\.output_log_dir=)[^\s]+')
if [ -z "$output_log_dir" ]; then
    output_log_dir=$(generate_output_log_dir "$input_string" "$model_name" "$scenario" "$gpu")
    input_string="$input_string harness_config.output_log_dir=${output_log_dir}"
fi
mkdir -p $output_log_dir

LOGFILE=$output_log_dir/output.log

# Apply power settings
source $SCRIPTS_DIR/power_settings.sh $model_name $gpu $scenario

# Looking for hangs
FILE_PIPE="$CODE_DIR/retcode_pipe"
# TODO: Consider moving this into the function
rm -f $FILE_PIPE && mkfifo $FILE_PIPE

# Start health checker
health_checker $FILE_PIPE $LOGFILE $model_name &
pid_health_checker=$!

# Kill processes
trap "pkill -P $pid_health_checker 2>/dev/null; kill $pid_health_checker 2>/dev/null" EXIT

# Main entry point of inference
python $CODE_DIR/main.py $input_string 2>&1 | tee $LOGFILE

# Check if the processes have been terminated or not
RETURN_CODE_MAIN=${PIPESTATUS[0]}
RETURN_CODE=$(timeout 2s bash -c "read val < $FILE_PIPE; echo \$val")

# Use the return code of main
if [[ -z $RETURN_CODE ]]; then
    RETURN_CODE=$RETURN_CODE_MAIN
fi

rm -f $FILE_PIPE
exit $RETURN_CODE
