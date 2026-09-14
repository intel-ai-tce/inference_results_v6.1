def parse_common_args(parser):
    parser.add_argument("-b", "--bucket", type=str, required=True, help="Name of the bucket")
    parser.add_argument("-a", "--aws_access_key_id", type=str, required=True)
    parser.add_argument("-s", "--aws_secret_access_key", type=str, required=True)
    parser.add_argument("-r", "--region_name", type=str, required=True)
    parser.add_argument("-e", "--endpoint_url", type=str, required=True)
    parser.add_argument("-p", "--remote_path", type=str)

    return parser

def split_s3_path(path):
    if not path.startswith("s3://"):
        raise ValueError(f"Invalid s3 path: {path}")
    parts = path[5:].split("/")
    bucket = parts[0]
    key = "/".join(parts[1:])
    return bucket, key


def dir_exists(path, s3_client):
    """Return True if the path is a 'directory'"""
    bucket, key = split_s3_path(path)

    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=key, Delimiter="/")
    if "CommonPrefixes" in response or "Contents" in response:
        return True
    return False


def list_objects(bucket, prefix, s3_client):
    """List objects in a prefix"""
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = response.get("Contents", [])
    while response.get("NextContinuationToken", None):
        response = s3_client.list_objects_v2(
            Bucket=bucket, Prefix=prefix, ContinuationToken=response["NextContinuationToken"]
        )
        keys += response["Contents"]
    return keys


def list_all_objects(path, s3_client):
    """List all objects based on the common prefixes"""
    bucket, key = split_s3_path(path)
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=key, Delimiter="/")
    prefix_map = {}

    if "CommonPrefixes" in response:
        for resp in response["CommonPrefixes"]:
            prefix = resp['Prefix']
            prefix_map[prefix] = list_objects(bucket, prefix, s3_client)
    elif "Contents" in response:
        prefix_map[key] = list_objects(bucket, key, s3_client)

    return prefix_map
