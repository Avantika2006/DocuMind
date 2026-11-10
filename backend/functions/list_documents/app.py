import json
import os
from decimal import Decimal

import boto3
from aws_lambda_powertools import Logger, Tracer

logger = Logger()
tracer = Tracer()
docs = boto3.resource('dynamodb').Table(os.environ['DOCS_TABLE'])


def to_json(o):
    if isinstance(o, Decimal):  # DynamoDB returns numbers as Decimal
        return int(o) if o % 1 == 0 else float(o)
    raise TypeError(f'Not serializable: {type(o)}')


@logger.inject_lambda_context
@tracer.capture_lambda_handler
def handler(event, context):
    items = docs.scan()['Items']  # fine for a demo-sized table; use a GSI at real scale
    items.sort(key=lambda d: d.get('createdAt', 0), reverse=True)
    return {'statusCode': 200, 'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'docs': items}, default=to_json)}