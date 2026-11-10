import io
import json
import os
import urllib.parse
from array import array

import boto3
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from aws_lambda_powertools import Logger, Metrics, Tracer
from aws_lambda_powertools.metrics import MetricUnit

logger = Logger()
tracer = Tracer()
metrics = Metrics()

s3 = boto3.client('s3')
ddb = boto3.resource('dynamodb')
docs = ddb.Table(os.environ['DOCS_TABLE'])
chunks = ddb.Table(os.environ['CHUNKS_TABLE'])

EMBED_PROVIDER = os.environ.get('EMBED_PROVIDER', 'none')   # none | bedrock
EMBED_MODEL = os.environ.get('EMBED_MODEL_ID', '')
bedrock = boto3.client('bedrock-runtime') if EMBED_PROVIDER == 'bedrock' else None

MAX_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_PAGES = 40
CHUNK_WORDS = 250
OVERLAP = 40


@tracer.capture_method
def embed(text):
    """Returns a vector, or None when embeddings are switched off."""
    if EMBED_PROVIDER == 'bedrock':
        body = json.dumps({'inputText': text, 'dimensions': 512, 'normalize': True})
        out = bedrock.invoke_model(modelId=EMBED_MODEL, body=body)
        return json.loads(out['body'].read())['embedding']
    return None


def chunk_page(text):
    """Split one page into overlapping windows of words."""
    words = text.split()
    step = CHUNK_WORDS - OVERLAP
    for i in range(0, len(words), step):
        yield ' '.join(words[i:i + CHUNK_WORDS])
        if i + CHUNK_WORDS >= len(words):
            break


def set_status(doc_id, status, **fields):
    fields['status'] = status
    docs.update_item(
        Key={'docId': doc_id},
        UpdateExpression='SET ' + ', '.join(f'#{k} = :{k}' for k in fields),
        ExpressionAttributeNames={f'#{k}': k for k in fields},
        ExpressionAttributeValues={f':{k}': v for k, v in fields.items()},
    )


@logger.inject_lambda_context
@tracer.capture_lambda_handler
@metrics.log_metrics
def handler(event, context):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(record['s3']['object']['key'])
        doc_id = key.split('/')[-1].removesuffix('.pdf')
        process(bucket, key, doc_id)


def process(bucket, key, doc_id):
    meta = docs.get_item(Key={'docId': doc_id}).get('Item')
    if meta and meta.get('status') == 'READY':
        logger.info('Already processed, skipping', extra={'docId': doc_id})  # idempotency
        return

    set_status(doc_id, 'PROCESSING')
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        if obj['ContentLength'] > MAX_BYTES:
            raise ValueError('File is larger than 10 MB')

        reader = PdfReader(io.BytesIO(obj['Body'].read()))
        pages = [reader.pages[i] for i in range(min(len(reader.pages), MAX_PAGES))]
        n = 0
        with chunks.batch_writer() as batch:
            for page_no, page in enumerate(pages, start=1):
                for piece in chunk_page(page.extract_text() or ''):
                    item = {'docId': doc_id, 'chunkId': f'{n:05d}', 'page': page_no, 'text': piece}
                    vec = embed(piece)
                    if vec is not None:
                        item['embedding'] = array('f', vec).tobytes()
                    batch.put_item(Item=item)
                    n += 1
        if n == 0:
            raise ValueError('No extractable text (is it a scanned PDF?)')

        set_status(doc_id, 'READY', pages=len(pages), chunks=n)
        metrics.add_metric(name='DocumentsIngested', unit=MetricUnit.Count, value=1)
        metrics.add_metric(name='ChunksCreated', unit=MetricUnit.Count, value=n)
        logger.info('Ingested', extra={'docId': doc_id, 'pages': len(pages), 'chunks': n})

    except (ValueError, PdfReadError) as e:
        # Bad input: retrying won't help
        logger.warning('Rejected document', extra={'docId': doc_id, 'reason': str(e)})
        set_status(doc_id, 'FAILED', error=str(e)[:200])
    except Exception as e:
        # Unknown/transient: re-raise so Lambda retries, then DLQ
        logger.exception('Ingest failed')
        set_status(doc_id, 'FAILED', error=str(e)[:200])
        raise