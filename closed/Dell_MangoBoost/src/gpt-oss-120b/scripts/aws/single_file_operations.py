import boto3

from argparse import ArgumentParser
from common import parse_common_args, split_s3_path
from pathlib import Path


def delete_file(bucket, object_to_del, s3_client):
    try:
        print(f"Try to delete in the bucket: {object_to_del}")
        s3_client.delete_object(Bucket=bucket, Key=object_to_del)
    except Exception as e:
        print(f"Error deleting file: {e}")


def download_file(bucket, args, s3_client):
    if not args.local_file:
        print('--local-file must be specified')
        return
    try:
        s3_client.download_file(bucket, args.remote_file, str(args.local_file))
        print(f"Downloaded {args.remote_file} as {args.local_file}")
    except Exception as e:
        print(f"Error downloading file: {e}")


def upload_file(bucket_name, args, s3_client):
    if not args.local_file or not args.local_file.is_file():
        print(f'--local_file is missing or not specified: {args.local_file}')
        return

    try:
        s3_client.upload_file(str(args.local_file), bucket_name, args.remote_file)
        print(f"File {args.local_file} uploaded to {args.remote_file} successfully.")
    except Exception as e:
        print(f"Error uploading file: {e}")


def main(args) -> None:
    boto_kwargs = {
        "aws_access_key_id": args.aws_access_key_id,
        "aws_secret_access_key": args.aws_secret_access_key,
        "region_name": args.region_name,
        "endpoint_url": args.endpoint_url,
    }

    s3_client = boto3.client("s3", **boto_kwargs)
    bucket_name, _ = split_s3_path(args.bucket)

    if args.command == 'download':
        download_file(bucket_name, args, s3_client)
    elif args.command == 'upload':
        upload_file(bucket_name, args, s3_client)
    elif args.command == 'delete':
        delete_file(bucket_name, args.remote_file, s3_client)
    else:
        print(f'Unknown command: {args.command}')
        return


if __name__ == "__main__":
    parser = ArgumentParser(description="Single file operations on the specified S3 bucket, download, upload and delete")
    parser = parse_common_args(parser)
    parser.add_argument("command", type=str, choices=['download', 'upload', 'delete'], help="Action to perform")
    parser.add_argument("--remote-file", type=str, required=True, help="File in the bucket")
    parser.add_argument("--local-file", type=Path, help="Local file")

    main(parser.parse_args())
