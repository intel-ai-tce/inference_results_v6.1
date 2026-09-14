import boto3
import concurrent.futures
import shutil

from argparse import ArgumentParser
from common import list_all_objects, parse_common_args, split_s3_path
from pathlib import Path


def _download_file_helper(s3_client, bucket, obj_key, dst_path):
    print(f'Downloading {obj_key} to {str(dst_path)}')
    s3_client.download_file(bucket, obj_key, str(dst_path))
    return dst_path.stat().st_size


def download_dir(args, s3_client, max_workers=16):
    remote_src = args.bucket
    local_dst = args.output_dir
    overwrite = args.overwrite

    max_workers = min([s3_client.meta.config.max_pool_connections, max_workers])
    print(f"Using {max_workers} workers")

    bucket, prefix = split_s3_path(remote_src)

    if overwrite:
        if local_dst.is_dir():
            shutil.rmtree(local_dst)

    objs = list_all_objects(remote_src, s3_client)
    if len(objs) == 0:
        print("Remote source is empty, nothing to do")
        return
    futures = []

    # Create a ThreadPoolExecutor with a specified number of max workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for obj_prefix, objects in objs.items():
            for obj in objects:
                dirname = obj_prefix.rstrip("/").split("/")[-1]
                dst_path = local_dst / dirname / obj["Key"][len(obj_prefix) :].lstrip("/")
                dst_path.parent.mkdir(parents=True, exist_ok=True)

                # Submit the download task to the executor
                futures.append(executor.submit(_download_file_helper, s3_client, bucket, obj["Key"], dst_path))


def list_files(bucket, s3_client):
    objects = list_all_objects(bucket, s3_client)
    if not objects:
        print("There are no files in the specified bucket")
        return

    object_count = 0
    for prefix, keys in objects.items():
        for key in keys:
            print(key['Key'])
            object_count += 1

    print(f"Found {object_count} files")


def main(args) -> None:
    boto_kwargs = {
        "aws_access_key_id": args.aws_access_key_id,
        "aws_secret_access_key": args.aws_secret_access_key,
        "region_name": args.region_name,
        "endpoint_url": args.endpoint_url,
    }

    s3_client = boto3.client("s3", **boto_kwargs)

    if args.remote_path:
        args.bucket = f"{args.bucket.strip('/ ')}/{args.remote_path}"

    # List the files in the bucket
    if args.list_files:
        list_files(args.bucket, s3_client)
        return

    # Download the files
    args.output_dir.mkdir(parents=True, exist_ok=True)
    download_dir(args, s3_client, max_workers=16)


if __name__ == "__main__":
    parser = ArgumentParser(description="Download files from S3 bucket with the provided credentials into a local folder")
    parser = parse_common_args(parser)
    parser.add_argument("-o", "--output_dir", type=Path, default='download', help="Directory where to download from the bucket")
    parser.add_argument("-l", "--list_files", action="store_true", help="List all the files in the specified bucket")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it is exist")
    args = parser.parse_args()

    main(args)
