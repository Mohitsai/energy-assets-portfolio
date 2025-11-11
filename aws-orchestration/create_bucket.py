# create_bucket.py
import boto3, botocore, os
s3 = boto3.client("s3")
bucket = os.environ.get("BUCKET", "energy-portfolio-data-agentic-trader")
region = boto3.session.Session().region_name or "us-east-1"
try:
    if region == "us-east-1":
        s3.create_bucket(Bucket=bucket)
    else:
        s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})
    print("created", bucket)
except botocore.exceptions.ClientError as e:
    print(e.response["Error"]["Message"])
