# 本文件定义代码向量生成清单及来源校验，独立于网络输入、检查点协议和粗哈希分类

import hashlib
import json
from pathlib import Path

from lua_semantic_source import text_sha256


CODE_EMBEDDINGS_META_FILENAME = 'code_embeddings_meta.json'
CODE_EMBEDDINGS_META_VERSION = 1
CODE_ENCODING_POLICY = {'version': 1, 'max_tokens': 256, 'overlap_tokens': 32,
                        'aggregation': 'new_token_weighted_mean_l2', 'blocks': 'source_role_name_order_v1'}
MAX_CODE_METADATA_BYTES = 128 * 1024 * 1024


def file_sha256(path):
    """流式校验资产身份，不把完整权重文件复制进内存"""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(effect):
    """由真实代码或保留的来源摘要获取身份，空/缺失代码不会伪装成正常文本"""
    code = effect.get('raw_code')
    provenance = effect.get('code_source', {})
    if not isinstance(provenance, dict):
        raise ValueError('invalid effect code provenance')
    if code is not None and not isinstance(code, str):
        raise ValueError('effect raw_code must be a string')
    if isinstance(code, str) and code.strip():
        digest = text_sha256(code)
        if provenance and provenance.get('sha256') != digest:
            raise ValueError('effect code source hash disagrees with raw_code')
        blocks = provenance.get('blocks', [])
        if not isinstance(blocks, list) or len(blocks) > 256:
            raise ValueError('invalid effect source blocks')
        previous = 0
        for block in blocks:
            if (not isinstance(block, dict) or block.get('role') not in ('context', 'registration', 'callback')
                    or not isinstance(block.get('name'), str) or len(block['name']) > 512
                    or type(block.get('start')) is not int or type(block.get('end')) is not int
                    or not previous <= block['start'] < block['end'] <= len(code)):
                raise ValueError('invalid effect source block')
            previous = block['end']
        encoded = hashlib.sha256(json.dumps({'source': digest, 'blocks': blocks}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return {'source_sha256': digest, 'encoding_sha256': encoded,
                'complete': bool(provenance.get('complete', False)), 'code': code, 'blocks': blocks}
    digest = provenance.get('sha256')
    if isinstance(digest, str) and len(digest) == 64 and all(char in '0123456789abcdef' for char in digest):
        # 清洁远程库可以去掉源码，但只能复用已验证的同来源向量，不能重新生成
        blocks = provenance.get('blocks', [])
        if not isinstance(blocks, list) or len(blocks) > 256:
            raise ValueError('invalid effect source blocks')
        previous = 0
        for block in blocks:
            if (not isinstance(block, dict) or block.get('role') not in ('context', 'registration', 'callback')
                    or not isinstance(block.get('name'), str) or len(block['name']) > 512
                    or type(block.get('start')) is not int or type(block.get('end')) is not int
                    or not previous <= block['start'] < block['end']):
                raise ValueError('invalid effect source block')
            previous = block['end']
        encoded = hashlib.sha256(json.dumps({'source': digest, 'blocks': blocks}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return {'source_sha256': digest, 'encoding_sha256': encoded,
                'complete': bool(provenance.get('complete', False)), 'code': None, 'blocks': blocks}
    raise ValueError('missing/empty effect code source; run local extraction before embedding')


def read_code_metadata(directory, index, shape, *, verify_files=True):
    """校验生成清单和向量/索引摘要；旧资产只返回未认证，不制造生成来源"""
    root = Path(directory).resolve()
    path = root / CODE_EMBEDDINGS_META_FILENAME
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CODE_METADATA_BYTES:
        raise ValueError('invalid code semantic metadata file')
    metadata = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(metadata, dict) or type(metadata.get('format_version')) is not int or metadata['format_version'] != CODE_EMBEDDINGS_META_VERSION:
        raise ValueError('unsupported code semantic metadata version')
    encoder, rows = metadata.get('encoder', {}), metadata.get('rows', {})
    if (not isinstance(encoder, dict) or not isinstance(encoder.get('name'), str)
            or not encoder['name'] or encoder.get('dimension') != shape[1]
            or not isinstance(encoder.get('fingerprint'), str) or len(encoder['fingerprint']) != 64
            or any(char not in '0123456789abcdef' for char in encoder['fingerprint'])
            or encoder.get('revision') is not None and not isinstance(encoder['revision'], str)
            or not isinstance(rows, dict) or set(rows) != set(index)
            or not isinstance(metadata.get('policy'), dict)):
        raise ValueError('code semantic metadata identity/rows mismatch')
    for record in rows.values():
        if not isinstance(record, dict):
            raise ValueError('invalid code semantic source record')
        for name in ('source_sha256', 'encoding_sha256'):
            value = record.get(name)
            if not isinstance(value, str) or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
                raise ValueError('invalid code semantic source digest')
        if (type(record.get('tokens')) is not int or record['tokens'] < 1
                or type(record.get('chunks')) is not int or record['chunks'] < 1
                or type(record.get('complete')) is not bool):
            raise ValueError('invalid code semantic coverage record')
    if verify_files:
        if not isinstance(metadata.get('files'), dict):
            raise ValueError('invalid code semantic file digest records')
        for filename in ('code_embeddings.npy', 'code_embeddings_idx.json'):
            if metadata.get('files', {}).get(filename) != file_sha256(root / filename):
                raise ValueError('code semantic provenance does not match embedding/index files')
    return metadata
