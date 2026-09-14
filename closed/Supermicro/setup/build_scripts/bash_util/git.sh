function get_revision() {
    if [ -z "$1" ]; then
        echo -e "${RED}Error: repository directory is not defined!${NC}"
        exit 1
    fi
    local repo_dir=$1
    echo $(git -C "$repo_dir" rev-parse --short HEAD)
}

function configure_git_repository() {
    if [ $# -ne 3 ]; then
        echo -e "${RED}Error: Expected 3 input parameters, but got $#.${NC}"
        exit 1
    fi

    local repo_url=$1
    local repo_revision=$2
    local repo_dir=$3

    git clone --filter=blob:none --recursive "$repo_url" "$repo_dir"

    git -C "$repo_dir" checkout $repo_revision
    git -C "$repo_dir" submodule sync
    git -C "$repo_dir" submodule update --init --recursive

    repo=$(basename $repo_dir)
    get_revision $repo_dir > "${repo_dir}/../${repo}_commit_hash.txt"
}

function apply_patch_file() {
    if [ $# -ne 2 ]; then
        echo -e "${RED}Error: Expected 2 input parameters, but got $#.${NC}"
        exit 1
    fi

    repo_patch_path=$1
    repo_dir=$2

    git -C "$repo_dir" apply "$repo_patch_path"

    git -C "$repo_dir" submodule sync 
    git -C "$repo_dir" submodule update --init --recursive
}
