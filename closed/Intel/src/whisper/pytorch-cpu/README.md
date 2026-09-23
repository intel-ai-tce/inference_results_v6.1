# Whisper CPU

## Build the RHAI 3.5 image

From this directory, run:

```bash
bash docker/build_container.sh
```

Override `IMAGE_NAME` to use a different image name or registry. The build
script explicitly selects `docker/Dockerfile.redhat`.
