import boto3

from argparse import ArgumentParser
from common import dir_exists, list_all_objects, parse_common_args, split_s3_path
from pathlib import Path


def upload_dir(bucket_dst, src, s3_client, overwrite=False):
    """
    Upload local directory to s3. If overwrite, removes contents of remote directory before uploading.

    src, bucket_dst: str
    s3_client:
    """
    src = Path(src)
    if not src.is_dir():
        raise ValueError("Source is not a directory")

    bucket, prefix = split_s3_path(bucket_dst)

    # If overwrite is True, remove the existing contents of the remote directory
    if overwrite:
        # List objects in the remote destination and delete them
        delete_dir(bucket_dst, s3_client)

    elif dir_exists(bucket_dst, s3_client):
        print(f"The path already exists: {prefix}")
        return

    print(f"Uploading content of {src}")

    # Upload files from src to the destination S3 bucket
    for path in src.glob("**/*"):
        if path.is_file():
            dest_key = prefix + "/" + str(path.relative_to(src))

            # Upload the file
            s3_client.upload_file(str(path), bucket, dest_key)


def delete_dir(bucket_dst, s3_client):
    objects = list_all_objects(bucket_dst, s3_client)
    bucket, prefix = split_s3_path(bucket_dst)

    for obj_prefix, keys in objects.items():
        for obj in keys:
            fn = obj["Key"]
            print(f"Deleting {fn}")
            s3_client.delete_object(Bucket=bucket, Key=fn)


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

    # Delete dir in S3 bucket without upload anything
    if args.delete:
        delete_dir(args.bucket, s3_client)
        return

    if not args.input_dir or not args.input_dir.is_dir():
        print('--input_dir is missing or not specified')
        return

    upload_dir(args.bucket, args.input_dir, s3_client, overwrite=args.overwrite)


if __name__ == "__main__":
    parser = ArgumentParser(description="Upload files from a local folder into the specified S3 bucket with the provided credentials")
    parser = parse_common_args(parser)
    parser.add_argument("-i", "--input_dir", type=Path, help="Directory from which files will be uploaded")
    parser.add_argument("--delete", action="store_true", help="Just delete files in the bucket")
    parser.add_argument("--overwrite", action="store_true", help="Delete files in the bucket, then upload the files from local directory")

    main(parser.parse_args())
