RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

set -e

#absolute path
BUILD_STEPS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
#relative path within docker build context
DOCKER_BUILD_CONTEXT=$( basename $( dirname  $BUILD_STEPS_DIR ) )
source "${BUILD_STEPS_DIR}/bash_util/git.sh"

function clean_up_build() {
    echo -e "${GREEN}Cleaning up build artifacts...${NC}"
    rm -rf $BUILD_STEPS_DIR/aiter_commit_hash.txt \
            $BUILD_STEPS_DIR/vllm_commit_hash.txt \
            $BUILD_STEPS_DIR/sglang_commit_hash.txt \
            $BUILD_STEPS_DIR/vllm \
            $BUILD_STEPS_DIR/aiter \
            $BUILD_STEPS_DIR/sglang \
            $BUILD_STEPS_DIR/BUILD_CONFIG \
            $BUILD_STEPS_DIR/generated_docker_image_name.txt \
            $BUILD_STEPS_DIR/mad_scripts \
            $BUILD_STEPS_DIR/dataset
}

function configure_vllm() {
    if [ -n "$VLLM_URL" ]; then
        echo -e "${GREEN}Setting up vLLM repository...${NC}"
        configure_git_repository "$VLLM_URL" "$VLLM_REV" "${BUILD_STEPS_DIR}/vllm"

        if [ -n "$VLLM_CUSTOM_PATCH" ]; then
            echo -e "${GREEN}Applying custom vLLM patch...${NC}"
            apply_patch_file "$SCRIPT_DIR/$VLLM_CUSTOM_PATCH" "${BUILD_STEPS_DIR}/vllm"
        fi
    else
        echo -e "${GREEN}Skipping setting up vLLM repository...${NC}"
    fi
}

function configure_aiter() {
    if [ -n "$AITER_URL" ]; then
        echo -e "${GREEN}Setting up AITER repository...${NC}"
        configure_git_repository "$AITER_URL" "$AITER_REV" "${BUILD_STEPS_DIR}/aiter"

        if [ -n "$AITER_CUSTOM_PATCH" ]; then    
            echo -e "${GREEN}Applying custom AITER patch...${NC}"
            apply_patch_file "$SCRIPT_DIR/$AITER_CUSTOM_PATCH" "${BUILD_STEPS_DIR}/aiter"
        fi

        if [ -n "$CK_CUSTOM_PATCH" ]; then   
            echo -e "${GREEN}Applying custom CK patch...${NC}"
            apply_patch_file "$SCRIPT_DIR/$CK_CUSTOM_PATCH" "${BUILD_STEPS_DIR}/aiter/3rdparty/composable_kernel"
        fi
    else
        echo -e "${GREEN}Skipping setting up AITER repository...${NC}"
    fi
}

function configure_sglang() {
    if [ -n "$SGLANG_URL" ]; then
        echo -e "${GREEN}Setting up SGLang repository...${NC}"
        configure_git_repository "$SGLANG_URL" "$SGLANG_REV" "${BUILD_STEPS_DIR}/sglang"
    else
        echo -e "${GREEN}Skipping setting up SGLang repository...${NC}"
    fi
}

function build_mlperf_dependencies() {
    if [ -z "$DOCKER_BASE_IMAGE" ]; then
        echo -e "${RED}Error: DOCKER_BASE_IMAGE is not defined!${NC}"
        exit 1
    fi

    if [ -z "$DOCKERFILE_MLPERF" ]; then
        echo -e "${RED}Error: DOCKERFILE_MLPERF is not defined!${NC}"
        exit 1
    fi

    if [ -z "$DOCKER_RESULT_IMAGE" ]; then
        echo -e "${RED}Error: DOCKER_RESULT_IMAGE is not defined!${NC}"
        exit 1
    fi

    echo -e "${GREEN}Building mlperf dependencies...${NC}"
    cp $SCRIPT_DIR/$CONFIG_FILE $BUILD_STEPS_DIR/BUILD_CONFIG
    docker build --build-arg BASE_IMAGE=${DOCKER_BASE_IMAGE} \
            --build-arg MLPERF_BUILD_CONFIG=$DOCKER_BUILD_CONTEXT/build_scripts/BUILD_CONFIG \
            -f "${BUILD_STEPS_DIR}/dockerfiles/$DOCKERFILE_MLPERF" \
            -t "${DOCKER_RESULT_IMAGE}_wip" "$SCRIPT_DIR/../.."
}

function adding_mad_scripts() {
    if [ -z "$DATASET_DIR" ]; then
        echo -e "${RED}Error: DATASET_DIR not defined!${NC}"
        exit 1
    fi

    if [ -z "$DOCKERFILE_MAD" ]; then
            echo -e "${RED}Error: DOCKERFILE_MAD not defined!${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Adding MAD scripts...${NC}"
    mkdir -p "${BUILD_STEPS_DIR}/mad_scripts"
    cp $SCRIPT_DIR/scripts/* "${BUILD_STEPS_DIR}/mad_scripts"
    echo -e "${GREEN}Adding dataset...${NC}"
    mkdir -p "${BUILD_STEPS_DIR}/dataset"
    cp -r $DATASET_DIR "${BUILD_STEPS_DIR}/dataset"
    docker build --build-arg BASE_IMAGE="$DOCKER_BASE_IMAGE" \
            --build-arg SCRIPTS_DIR="$DOCKER_BUILD_CONTEXT/build_scripts/mad_scripts" \
            --build-arg DATASET_DIR="$DOCKER_BUILD_CONTEXT/build_scripts/dataset" \
            -f "${BUILD_STEPS_DIR}/dockerfiles/$DOCKERFILE_MAD" \
            -t "${DOCKER_RESULT_IMAGE}" "$SCRIPT_DIR/../.."
}

function reinstall_vllm() {
    if [ -n "$VLLM_URL" ]; then
        echo -e "${GREEN}Reinstalling vLLM...${NC}"
        docker build --build-arg BASE_IMAGE="${DOCKER_RESULT_IMAGE}_wip" \
                --build-arg VLLM_DIR="$DOCKER_BUILD_CONTEXT/build_scripts/vllm" \
                --build-arg GPU_ARCH="$GPU_ARCH" \
                -f "${BUILD_STEPS_DIR}/dockerfiles/$DOCKERFILE_VLLM" \
                -t "${DOCKER_RESULT_IMAGE}_wip" "$SCRIPT_DIR/../.."
    else
        echo -e "${GREEN}Skipping reinstallation of vLLM...${NC}"
    fi
}

function reinstall_aiter() {
    if [ -n "$AITER_URL" ]; then
        echo -e "${GREEN}Reinstalling aiter...${NC}"
        docker build --build-arg BASE_IMAGE="${DOCKER_RESULT_IMAGE}_wip" \
                --build-arg AITER_DIR="$DOCKER_BUILD_CONTEXT/build_scripts/aiter" \
                --build-arg GPU_ARCH="${GPU_ARCH//,/;}" \
                -f "${BUILD_STEPS_DIR}/dockerfiles/$DOCKERFILE_AITER" \
                -t "${DOCKER_RESULT_IMAGE}_wip" "$SCRIPT_DIR/../.."
    else
        echo -e "${GREEN}Skipping reinstallation of aiter...${NC}"
    fi
}

function reinstall_sglang() {
    if [ -n "$SGLANG_URL" ]; then
        echo -e "${GREEN}Reinstalling SGLang...${NC}"
        local BUILD_ARGS="--build-arg BASE_IMAGE=${DOCKER_RESULT_IMAGE}_wip \
                        --build-arg SGLANG_DIR=$DOCKER_BUILD_CONTEXT/build_scripts/sglang"
        if [ -n "$GPU_ARCH" ]; then
            BUILD_ARGS="$BUILD_ARGS --build-arg GPU_ARCH=$GPU_ARCH";
        fi
        if [ -n "$SGLANG_BUILD_FLAGS" ]; then
            BUILD_ARGS="$BUILD_ARGS --build-arg BUILD_FLAGS=$SGLANG_BUILD_FLAGS";
        fi
        docker build $BUILD_ARGS \
                -f "${BUILD_STEPS_DIR}/dockerfiles/$DOCKERFILE_SGLANG" \
                -t "${DOCKER_RESULT_IMAGE}_wip" "$SCRIPT_DIR/../.."
    else
        echo -e "${GREEN}Skipping reinstallation of SGLang...${NC}"
    fi
}

function patch_sglang() {
    if [ -n "$SGLANG_LOCATION_DIR" ]; then
        if [ -z "$SGLANG_CUSTOM_PATCH" ]; then
            echo -e "${RED}Error: SGLANG_CUSTOM_PATCH not defined!${NC}"
            exit 1
        fi

        echo -e "${GREEN}Patching SGLang...${NC}"
        docker build --build-arg BASE_IMAGE="${DOCKER_RESULT_IMAGE}_wip" \
                --build-arg SGLANG_DIR="$SGLANG_LOCATION_DIR" \
                --build-arg PATCH_FILE="$DOCKER_BUILD_CONTEXT/$SGLANG_CUSTOM_PATCH" \
                -f "${BUILD_STEPS_DIR}/dockerfiles/$DOCKERFILE_SGLANG" \
                -t "${DOCKER_RESULT_IMAGE}_wip" "$SCRIPT_DIR/../.."
    else
        echo -e "${GREEN}Skipping patching SGLang...${NC}"
    fi
}

function preload_triton_kernels() {
    if [ -n "$PRELOAD_TRITON_KERNELS" ]; then
        echo -e "${GREEN}Preloading triton kernels...${NC}"

        CONTAINER_NAME="mlperf_kernel_preload_$$"
        CODE_DIR=$( dirname $( dirname "$BUILD_STEPS_DIR" ) )/code

        docker run -d --name "$CONTAINER_NAME" \
            --device=/dev/kfd --device=/dev/dri \
            --security-opt seccomp=unconfined \
            --group-add video \
            -v ${CODE_DIR}:/lab-mlperf-inference/code \
            "${DOCKER_RESULT_IMAGE}_wip" \
            sleep infinity

        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to create container for kernel preloading!${NC}"
            exit 1
        fi

        echo -e "${YELLOW}Running kernel preload script...${NC}"
        docker exec "$CONTAINER_NAME" python3 /lab-mlperf-inference/code/harness_llm/common/preload/triton_preload.py

        if [ $? -ne 0 ]; then
            echo -e "${RED}Kernel preloading script failed!${NC}"
            docker rm -f "$CONTAINER_NAME"
            echo -e "${RED}Kernel preloading failed!${NC}"
            exit 1
        fi

        echo -e "${YELLOW}Saving preloaded kernels to image...${NC}"
        docker commit --change='CMD ["/bin/bash"]' "$CONTAINER_NAME" "${DOCKER_RESULT_IMAGE}_wip"

        if [ $? -ne 0 ]; then
            docker rm -f "$CONTAINER_NAME"
            echo -e "${RED}Failed to commit container to image!${NC}"
            exit 1
        fi

        docker rm -f "$CONTAINER_NAME"

        echo -e "${GREEN}Kernels preloaded successfully${NC}"
    else
        echo -e "${GREEN}Skipping preloading triton kernels...${NC}"
    fi
}

function preload_aiter_kernels() {
    if [ -n "$PRELOAD_AITER_COMMAND_ARGS" ]; then
        echo -e "${GREEN}Preloading aiter kernels...${NC}"

        if [[ ! "$PRELOAD_AITER_COMMAND_ARGS" == *"--backend vllm"* && ! "$PRELOAD_AITER_COMMAND_ARGS" == *"--backend sglang"* ]]; then
            echo -e "${YELLOW}Preloading aiter kernels only supported by vllm and sglang backend! ${NC}"
            echo -e "${YELLOW}Skipping preloading aiter kernels...${NC}"
            return 0
        fi

        CONTAINER_NAME="mlperf_kernel_preload_$$"
        CODE_DIR=$( dirname $( dirname "$BUILD_STEPS_DIR" ) )/code
        PRELOAD_AITER_MODEL_DIR=${PRELOAD_AITER_MODEL_DIR:-"/data/inference/model/"}

        docker run -d --name "$CONTAINER_NAME" \
            --ipc=host --network=host --privileged \
            --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
            --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
            --security-opt seccomp=unconfined \
            --group-add video \
            -v ${CODE_DIR}:/lab-mlperf-inference/code \
            -v ${PRELOAD_AITER_MODEL_DIR}:/model/ \
            "${DOCKER_RESULT_IMAGE}_wip" \
            sleep infinity

        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to create container for kernel preloading!${NC}"
            exit 1
        fi

        echo -e "${YELLOW}Running kernel preload script...${NC}"
        docker exec "$CONTAINER_NAME" python3 /lab-mlperf-inference/code/preload.py $PRELOAD_AITER_COMMAND_ARGS #don't use quoted "$PRELOAD_AITER_COMMAND_ARGS"

        if [ $? -ne 0 ]; then
            echo -e "${RED}Kernel preloading script failed!${NC}"
            docker rm -f "$CONTAINER_NAME"
            echo -e "${RED}Kernel preloading failed!${NC}"
            exit 1
        fi

        echo -e "${YELLOW}Saving preloaded kernels to image...${NC}"
        docker commit --change='CMD ["/bin/bash"]' "$CONTAINER_NAME" "${DOCKER_RESULT_IMAGE}_wip"

        if [ $? -ne 0 ]; then
            docker rm -f "$CONTAINER_NAME"
            echo -e "${RED}Failed to commit container to image!${NC}"
            exit 1
        fi

        docker rm -f "$CONTAINER_NAME"

        echo -e "${GREEN}Kernels preloaded successfully${NC}"
    else
        echo -e "${GREEN}Skipping preloading aiter kernels...${NC}"
    fi
}

function tag_docker_image() {
    echo -e "${GREEN}Naming docker image...${NC}"
    
    local user_repo=rocm/mlperf-inference
    local harness_revision=_h_$( get_revision $PWD )
    local aiter_revision=$( cat $BUILD_STEPS_DIR/aiter_commit_hash.txt 2> /dev/null )
    [[ -n "$aiter_revision" ]] && aiter_revision=_a_$aiter_revision
    local vllm_revision=$( cat $BUILD_STEPS_DIR/vllm_commit_hash.txt 2> /dev/null )
    [[ -n "$vllm_revision" ]] && vllm_revision=_v_$vllm_revision
    local sglang_revision=$( cat $BUILD_STEPS_DIR/sglang_commit_hash.txt 2> /dev/null )
    [[ -n "$sglang_revision" ]] && sglang_revision=_s_$sglang_revision
    local build_config_hash=_b_$(md5sum "$SCRIPT_DIR/$CONFIG_FILE" | cut -c1-8)
    local base_image_tag=${DOCKER_BASE_IMAGE##*:}
    local generated_image_name=${user_repo}:${base_image_tag}_${harness_revision:-}${vllm_revision}${sglang_revision}${aiter_revision}${build_config_hash}
    echo $generated_image_name > $BUILD_STEPS_DIR/generated_docker_image_name.txt
    
    docker tag ${DOCKER_RESULT_IMAGE}_wip $generated_image_name
    docker tag $generated_image_name  ${DOCKER_RESULT_IMAGE}

    echo "Generated image name: $generated_image_name"
    # echo "Result image name: $DOCKER_RESULT_IMAGE"
}

function push_docker_result() {
    if [ -n "$DOCKER_PUSH_RESULT" ]; then
        echo -e "${GREEN}Pushing docker images...${NC}"
        local generated_image_name=$( cat  $BUILD_STEPS_DIR/generated_docker_image_name.txt )

        docker push $DOCKER_RESULT_IMAGE
        docker push $generated_image_name
    fi
}

function build_dataset_and_model_dependencies() {

    if [ -z "$DOCKER_BASE_IMAGE" ]; then
        echo -e "${RED}Error: DOCKER_BASE_IMAGE is not defined!${NC}"
        exit 1
    fi

    if [ -z "$DOCKERFILE_DATASET_AND_MODEL" ]; then
        echo -e "${RED}Error: DOCKERFILE_DATASET_AND_MODEL is not defined!${NC}"
        exit 1
    fi

    if [ -z "$DOCKER_RESULT_IMAGE" ]; then
        echo -e "${RED}Error: DOCKER_RESULT_IMAGE is not defined!${NC}"
        exit 1
    fi

    if [ -z "$(docker images -q $DOCKER_RESULT_IMAGE)" ]; then
        echo -e "${GREEN}Building dataset and model dependencies...${NC}"

        cp $BUILD_STEPS_DIR/$CONFIG_FILE $BUILD_STEPS_DIR/BUILD_CONFIG
        docker build --no-cache --build-arg BASE_IMAGE=${DOCKER_BASE_IMAGE} \
                --build-arg MLPERF_BUILD_CONFIG=$DOCKER_BUILD_CONTEXT/build_scripts/BUILD_CONFIG ${DOCKER_BUILD_EXTRA_ARGS} \
                -f "${BUILD_STEPS_DIR}/dockerfiles/$DOCKERFILE_DATASET_AND_MODEL" \
                -t ${DOCKER_RESULT_IMAGE} "$SCRIPT_DIR/../.."
    else
        echo -e "${GREEN}Docker image ${YELLOW}${DOCKER_RESULT_IMAGE}${GREEN} already exists, skipping build...${NC}"
    fi
}
