import boto3
import re

def handler(event, context):
    s3_client = boto3.client('s3')
    model_uri = event['s3_model_artifact']
    bucket_name, full_path = model_uri.replace("s3://", "").split("/", 1)
    pattern = r'pipelines-[^/]*'
    match = re.search(pattern, full_path)

    if match:
        folder_path = match.group(0)
        prefix = f"{full_path.split(folder_path)[0]}{folder_path}/"
    else:
        return {
            'statusCode': 400,
            'body': 'No valid pipeline folder found in the S3 URI.'
        }

    try:
        # List all objects in the folder
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        to_delete = [{'Key': obj['Key']} for obj in response.get('Contents', [])]

        # Delete all objects found
        if to_delete:
            delete_response = s3_client.delete_objects(
                Bucket=bucket_name,
                Delete={'Objects': to_delete}
            )
            return {
                'statusCode': 200,
                'body': f'Successfully deleted folder {prefix} from S3.'
            }
        else:
            return {
                'statusCode': 404,
                'body': 'No objects found to delete.'
            }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': f'Error deleting folder {prefix}: {str(e)}'
        }
