import json
import os
import time
import uuid

import boto3
from botocore.config import Config
from aws_lambda_powertools import Logger, Tracer

logger = Logger()
tracer = Tracer()

s3 = boto3.client('s3', config=Config(signature_version='s3v4'))
docs = boto3.resource('dynamodb').Table(os.environ['DOCS_TABLE'])
BUCKET = os.environ['BUCKET']


def resp(code, body):
    return {'statusCode': code, 'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(body)}


@logger.inject_lambda_context
@tracer.capture_lambda_handler
def handler(event, context):
    params = event.get('queryStringParameters') or {}
    filename = params.get('filename', 'document.pdf')
    if not filename.lower().endswith('.pdf'):
        return resp(400, {'error': 'Only PDF files are supported'})

    doc_id = uuid.uuid4().hex[:12]
    key = f'docs/{doc_id}.pdf'
    upload_url = s3.generate_presigned_url(
        'put_object',
        Params={'Bucket': BUCKET, 'Key': key, 'ContentType': 'application/pdf'},
        ExpiresIn=300,  # 5 minutes
    )
    docs.put_item(Item={
        'docId': doc_id,
        'filename': filename[:200],
        'status': 'UPLOADING',
        'createdAt': int(time.time()),
    })
    logger.info('Issued upload URL', extra={'docId': doc_id})
    return resp(200, {'docId': doc_id, 'uploadUrl': upload_url})