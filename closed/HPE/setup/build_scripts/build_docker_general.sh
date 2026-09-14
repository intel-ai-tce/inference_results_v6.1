source "$SCRIPT_DIR/../build_scripts/build_steps.sh"

function build() {
    clean_up_build
    configure_vllm
    configure_aiter
    configure_sglang
    build_mlperf_dependencies
    reinstall_sglang
    patch_sglang
    reinstall_vllm
    reinstall_aiter
    preload_triton_kernels
    preload_aiter_kernels
    tag_docker_image
    push_docker_result
    clean_up_build
}

function build_mad() {
    clean_up_build
    adding_mad_scripts
    push_docker_result
    clean_up_build
}

function build_dataset_and_model() {
    clean_up_build
    build_dataset_and_model_dependencies
    push_docker_result    
    clean_up_build
}
