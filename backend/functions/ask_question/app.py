import json
import math
import os
import re
import time
import urllib.request
from array import array
from collections import Counter

import boto3
from boto3.dynamodb.conditions import Key
from aws_lambda_powertools import Logger, Metrics, Tracer
from aws_lambda_powertools.metrics import MetricUnit

logger = Logger()
tracer = Tracer()
metrics = Metrics()

ddb = boto3.resource('dynamodb')
docs = ddb.Table(os.environ['DOCS_TABLE'])
chunks = ddb.Table(os.environ['CHUNKS_TABLE'])

LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'none')   # none | gemini | bedrock
LLM_MODEL = os.environ.get('LLM_MODEL_ID', '')
EMBED_MODEL = os.environ.get('EMBED_MODEL_ID', '')
GEMINI_KEY_PARAM = os.environ.get('GEMINI_KEY_PARAM', '')
TOP_K = 4

bedrock = boto3.client('bedrock-runtime') if LLM_PROVIDER == 'bedrock' else None
_gemini_key = None  # cached across warm invocations

SYSTEM_PROMPT = (
    'Answer the question using ONLY the document excerpts provided. '
    'After each claim, cite the page like [p. 3]. '
    'If the excerpts do not contain the answer, say you could not find it in the document.'
)
STOPWORDS = set(
    'a an and are as at be by can do does for from has have how i in is it its of on or '
    'that the their this to was were what when where which who why will with you your'.split()
)


def resp(code, body):
    return {'statusCode': code, 'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(body)}


# ---------- Retrieval ----------
def tokenize(text):
    return [w for w in re.findall(r'[a-z0-9]+', text.lower()) if w not in STOPWORDS]


def bm25_rank(question, items, k1=1.5, b=0.75):
    toks = [tokenize(c['text']) for c in items]
    n = len(toks)
    avgdl = sum(len(t) for t in toks) / max(n, 1)
    df = Counter(term for t in toks for term in set(t))
    q_terms = set(tokenize(question))
    scored = []
    for t, item in zip(toks, items):
        tf = Counter(t)
        score = 0.0
        for term in q_terms:
            if term in tf:
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(t) / avgdl))
        scored.append((score, item))
    return sorted(scored, key=lambda s: s[0], reverse=True)


def vector_rank(question, items):
    body = json.dumps({'inputText': question, 'dimensions': 512, 'normalize': True})
    q = json.loads(bedrock.invoke_model(modelId=EMBED_MODEL, body=body)['body'].read())['embedding']
    scored = []
    for c in items:
        v = array('f')
        v.frombytes(c['embedding'].value)
        scored.append((sum(x * y for x, y in zip(q, v)), c))
    return sorted(scored, key=lambda s: s[0], reverse=True)


@tracer.capture_method
def load_chunks(doc_id):
    items, kwargs = [], {'KeyConditionExpression': Key('docId').eq(doc_id)}
    while True:
        page = chunks.query(**kwargs)
        items.extend(page['Items'])
        if 'LastEvaluatedKey' not in page:
            return items
        kwargs['ExclusiveStartKey'] = page['LastEvaluatedKey']


# ---------- Generation ----------
def gemini_key():
    global _gemini_key
    if _gemini_key is None:
        ssm = boto3.client('ssm')
        _gemini_key = ssm.get_parameter(Name=GEMINI_KEY_PARAM, WithDecryption=True)['Parameter']['Value']
    return _gemini_key


def extractive(top):
    _, best = top[0]
    return f'Most relevant passage [p. {int(best["page"])}]:\n{best["text"]}'


@tracer.capture_method
def generate(question, top):
    excerpts = '\n\n'.join(f'[p. {int(c["page"])}] {c["text"]}' for _, c in top)
    prompt = f'Excerpts:\n{excerpts}\n\nQuestion: {question}'

    if LLM_PROVIDER == 'bedrock':
        out = bedrock.converse(
            modelId=LLM_MODEL,
            system=[{'text': SYSTEM_PROMPT}],
            messages=[{'role': 'user', 'content': [{'text': prompt}]}],
            inferenceConfig={'maxTokens': 600, 'temperature': 0.2},
        )
        return out['output']['message']['content'][0]['text']

    if LLM_PROVIDER == 'gemini':
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent'
        payload = {
            'systemInstruction': {'parts': [{'text': SYSTEM_PROMPT}]},
            'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
            'generationConfig': {'temperature': 0.2, 'maxOutputTokens': 2048},
        }
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), method='POST',
            headers={'Content-Type': 'application/json', 'x-goog-api-key': gemini_key()},
        )
        with urllib.request.urlopen(req, timeout=25) as r:
            data = json.loads(r.read())
        parts = data['candidates'][0]['content']['parts']
        return ''.join(p.get('text', '') for p in parts)

    return extractive(top)


# ---------- Handler ----------
@logger.inject_lambda_context
@tracer.capture_lambda_handler
@metrics.log_metrics
def handler(event, context):
    started = time.time()
    try:
        body = json.loads(event.get('body') or '{}')
    except json.JSONDecodeError:
        return resp(400, {'error': 'Body must be JSON'})

    doc_id = body.get('docId')
    question = (body.get('question') or '').strip()
    if not doc_id or not question:
        return resp(400, {'error': 'docId and question are required'})
    if len(question) > 500:
        return resp(400, {'error': 'Question too long (max 500 characters)'})

    meta = docs.get_item(Key={'docId': doc_id}).get('Item')
    if not meta or meta['status'] != 'READY':
        return resp(409, {'error': 'Document is not ready yet'})

    items = load_chunks(doc_id)
    use_vectors = bedrock is not None and all('embedding' in c for c in items)
    ranked = vector_rank(question, items) if use_vectors else bm25_rank(question, items)
    top = ranked[:TOP_K]
    retrieval = 'vector' if use_vectors else 'bm25'

    if not top or top[0][0] <= 0:
        answer, mode = 'I could not find anything about that in the document.', 'no-match'
    else:
        try:
            answer, mode = generate(question, top), LLM_PROVIDER
        except Exception:
            # Graceful degradation: LLM down or rate-limited -> still answer
            logger.exception('LLM call failed, falling back to extractive answer')
            answer, mode = extractive(top), 'extractive-fallback'

    latency_ms = int((time.time() - started) * 1000)
    metrics.add_metric(name='QuestionsAnswered', unit=MetricUnit.Count, value=1)
    metrics.add_metric(name='AnswerLatency', unit=MetricUnit.Milliseconds, value=latency_ms)
    logger.info('Answered', extra={'docId': doc_id, 'mode': mode, 'retrieval': retrieval, 'latencyMs': latency_ms})

    return resp(200, {
        'answer': answer,
        'mode': mode,
        'retrieval': retrieval,
        'latencyMs': latency_ms,
        'sources': [{'page': int(c['page']), 'score': round(s, 3), 'snippet': c['text'][:200]}
                    for s, c in top],
    })