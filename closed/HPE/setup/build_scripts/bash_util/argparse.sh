#!/usr/bin/env bash

# Reusable Bash Argument Parser Library
# Source this file to use the parse_args function in your scripts
#
# Usage:
#   source argparse.sh
#   
#   # Define your arguments
#   declare -A ARG_SPECS=(
#       ["--token"]="value required TOKEN"
#       ["--skip-download"]="flag"
#       ["--type"]="value required TYPE choices:A,B"
#       ["--verbose"]="flag"
#       ["--output"]="value optional OUTPUT default:output.txt"
#   )
#   
#   # Parse arguments
#   parse_args "$@"
#   
#   # Access parsed values
#   echo "Token: ${ARGS[--token]}"
#   if [[ "${ARGS[--skip-download]}" == "true" ]]; then
#       echo "Skipping download"
#   fi

# Global associative array to store parsed arguments
declare -gA ARGS

# Print error message and exit
_argparse_error() {
    echo "ERROR: $*" >&2
    exit 1
}

# Print usage information based on ARG_SPECS
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    
    for arg_name in "${!ARG_SPECS[@]}"; do
        local spec="${ARG_SPECS[$arg_name]}"
        local description=""
        
        if [[ "$spec" == *"flag"* ]]; then
            description="$arg_name                 (flag)"
        elif [[ "$spec" == *"required"* ]]; then
            local var_name=$(echo "$spec" | grep -oP '(?<=required )[A-Z_]+')
            description="$arg_name <$var_name>    (required)"
        else
            local var_name=$(echo "$spec" | grep -oP '(?<=optional )[A-Z_]+')
            description="$arg_name <$var_name>    (optional)"
        fi
        
        # Add choices if specified
        if [[ "$spec" == *"choices:"* ]]; then
            local choices=$(echo "$spec" | grep -oP '(?<=choices:)[^ ]+')
            description="$description [choices: $choices]"
        fi
        
        # Add default if specified
        if [[ "$spec" == *"default:"* ]]; then
            local default=$(echo "$spec" | grep -oP '(?<=default:)[^ ]+')
            description="$description [default: $default]"
        fi
        
        echo "  $description"
    done
    
    echo ""
}

# Validate that required arguments are present
_validate_required_args() {
    local missing_args=()
    
    for arg_name in "${!ARG_SPECS[@]}"; do
        local spec="${ARG_SPECS[$arg_name]}"
        
        if [[ "$spec" == *"required"* ]] && [[ -z "${ARGS[$arg_name]}" ]]; then
            missing_args+=("$arg_name")
        fi
    done
    
    if [[ ${#missing_args[@]} -gt 0 ]]; then
        _argparse_error "Missing required arguments: ${missing_args[*]}"
    fi
}

# Validate argument value against choices
_validate_choices() {
    local arg_name="$1"
    local value="$2"
    local spec="${ARG_SPECS[$arg_name]}"
    
    if [[ "$spec" == *"choices:"* ]]; then
        local choices=$(echo "$spec" | grep -oP '(?<=choices:)[^ ]+')
        IFS=',' read -ra choice_array <<< "$choices"
        
        local valid=false
        for choice in "${choice_array[@]}"; do
            if [[ "$value" == "$choice" ]]; then
                valid=true
                break
            fi
        done
        
        if [[ "$valid" == "false" ]]; then
            _argparse_error "Invalid value '$value' for $arg_name. Valid choices: $choices"
        fi
    fi
}

# Set default values for optional arguments
_set_defaults() {
    for arg_name in "${!ARG_SPECS[@]}"; do
        local spec="${ARG_SPECS[$arg_name]}"
        
        if [[ "$spec" == *"default:"* ]] && [[ -z "${ARGS[$arg_name]}" ]]; then
            local default=$(echo "$spec" | grep -oP '(?<=default:)[^ ]+')
            ARGS[$arg_name]="$default"
        fi
    done
}

# Main argument parsing function
parse_args() {
    # Initialize ARGS array
    ARGS=()
    
    # Check for --help or -h
    for arg in "$@"; do
        if [[ "$arg" == "--help" ]] || [[ "$arg" == "-h" ]]; then
            print_usage
            exit 0
        fi
    done
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        local key="$1"
        
        # Check if argument is defined in ARG_SPECS
        if [[ -z "${ARG_SPECS[$key]}" ]]; then
            _argparse_error "Unknown argument: $key"
        fi
        
        local spec="${ARG_SPECS[$key]}"
        
        if [[ "$spec" == *"flag"* ]]; then
            # Handle flag (boolean) arguments
            ARGS[$key]="true"
            shift
        elif [[ "$spec" == *"value"* ]]; then
            # Handle arguments that take a value
            shift
            if [[ $# -eq 0 ]] || [[ "$1" == --* ]]; then
                _argparse_error "$key requires a value"
            fi
            
            local value="$1"
            _validate_choices "$key" "$value"
            ARGS[$key]="$value"
            shift
        else
            _argparse_error "Invalid spec for $key: $spec"
        fi
    done
    
    # Set defaults for optional arguments
    _set_defaults
    
    # Validate that all required arguments are present
    _validate_required_args
}

# Helper function to check if a flag is set
has_flag() {
    local flag="$1"
    [[ "${ARGS[$flag]}" == "true" ]]
}

# Helper function to check if a flag is set
get_flag() {
    local flag="$1"
    local default="false"
    echo "${ARGS[$flag]:-$default}"
}

# Helper function to get argument value with fallback
get_arg() {
    local arg_name="$1"
    local default="${2:-}"
    echo "${ARGS[$arg_name]:-$default}"
}
