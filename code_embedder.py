# 本文件按真实Lua代码内容接续生成固定384维语义向量，并保存可追溯生成清单
import hashlib
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import os
import tempfile
from pathlib import Path

from semantic_assets import (
    build_expected_code_semantic_keys,
    CODE_EMBEDDINGS_FILENAME,
    CODE_EMBEDDINGS_INDEX_FILENAME,
    invalidate_static_semantic_assets,
    validate_code_semantic_assets,
    validate_semantic_bundle,
)
from code_semantic_provenance import (
    CODE_EMBEDDINGS_META_FILENAME, CODE_EMBEDDINGS_META_VERSION, CODE_ENCODING_POLICY,
    file_sha256, source_identity,
)

class CodeSemanticEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2', revision=None):
        """初始化离线资产生成器；编码器修订不属于决斗网络结构"""
        self.model_name = model_name
        self.revision = revision
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None

    def _get_model(self):
        """仅在确实需要计算新向量时加载代码语义模型"""
        if self.model is None:
            print(f"[代码语义] 正在加载离线语义编码器 {self.model_name}...")
            self.model = SentenceTransformer(self.model_name, device=self.device, revision=getattr(self, '_pinned_revision', getattr(self, 'revision', None)))
        return self.model

    def _get_expected_dimension(self):
        """默认模型免加载返回维度，自定义模型则读取实际配置"""
        if self.model is not None:
            return int(self.model.get_sentence_embedding_dimension())
        if self.model_name == 'all-MiniLM-L6-v2':
            return 384
        return int(self._get_model().get_sentence_embedding_dimension())

    def _collect_effect_code(self, knowledge_base):
        """按卡号和效果槽稳定收集需要向量化的 Lua 代码"""
        entries = []
        for card_id in sorted(knowledge_base, key=lambda value: int(value)):
            data = knowledge_base[card_id]
            effects = sorted(
                data.get('effects', []),
                key=lambda effect: int(effect.get('slot', 1) or 1),
            )
            for effect in effects:
                slot_idx = int(effect.get('slot', 1) or 1) - 1
                if not 0 <= slot_idx < 8:
                    continue
                entries.append((f"{card_id}_{slot_idx}", source_identity(effect)))
        return entries

    def _encoder_identity(self):
        """记录实际模型权重/配置身份，维度相同不能代表编码器相同"""
        model = self._get_model()
        digest = hashlib.sha256()
        digest.update(str(type(model).__module__ + '.' + type(model).__qualname__).encode())
        digest.update(str(model.get_sentence_embedding_dimension()).encode())
        if hasattr(model, 'state_dict'):
            for name, module in model._modules.items():
                digest.update((name + ':' + type(module).__module__ + '.' + type(module).__qualname__).encode())
                if hasattr(module, 'get_config_dict'):
                    digest.update(json.dumps(module.get_config_dict(), sort_keys=True, default=str).encode())
            for name, tensor in sorted(model.state_dict().items()):
                value = tensor.detach().cpu().contiguous()
                digest.update(name.encode())
                digest.update(str(value.dtype).encode())
                digest.update(value.numpy().tobytes())
        else:
            # 单元测试的小型编码器也必须有实现身份，不能只凭输出宽度复用
            digest.update(getattr(type(model).encode, '__code__').co_code)
        tokenizer = getattr(model, 'tokenizer', None)
        if tokenizer is not None:
            digest.update(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode())
            digest.update(str(getattr(tokenizer, 'do_lower_case', None)).encode())
            if hasattr(tokenizer, 'backend_tokenizer'):
                # 编码调用会临时修改padding/truncation，不能把运行状态当成模型更新
                configuration = json.loads(tokenizer.backend_tokenizer.to_str())
                configuration['padding'] = None
                configuration['truncation'] = None
                digest.update(json.dumps(configuration, ensure_ascii=False, separators=(',', ':')).encode())
        revision = None
        if hasattr(model, '_first_module'):
            first = model._first_module()
            config = getattr(getattr(first, 'auto_model', None), 'config', None)
            revision = getattr(config, '_commit_hash', None)
            if config is not None:
                configuration = config.to_dict()
                configuration.pop('_name_or_path', None)
                configuration.pop('_commit_hash', None)
                digest.update(json.dumps(configuration, sort_keys=True).encode())
        return {'name': self.model_name, 'revision': revision or getattr(self, '_pinned_revision', None) or getattr(self, 'revision', None),
                'dimension': self._get_expected_dimension(), 'fingerprint': digest.hexdigest()}

    def _split_chunks(self, entry):
        """按源码块分窗覆盖全部token，保留角色/顺序提示且不静默截断尾部"""
        model = self._get_model()
        tokenizer = getattr(model, 'tokenizer', None)
        code, blocks = entry['code'], entry['blocks']
        if code is None:
            raise ValueError('code source was removed; cannot regenerate this vector')
        if tokenizer is None:
            if len(code.split()) > CODE_ENCODING_POLICY['max_tokens']:
                raise ValueError('embedding tokenizer is required for long-code coverage')
            return [(code, max(1, len(code.split())))], max(1, len(code.split()))
        maximum = min(CODE_ENCODING_POLICY['max_tokens'], int(model.max_seq_length))
        if maximum < 64:
            raise ValueError('code encoder context is too short')
        pieces, token_count = [], 0
        blocks = blocks or [{'role': 'callback', 'name': '', 'start': 0, 'end': len(code)}]
        for ordinal, block in enumerate(blocks):
            body = code[block['start']:block['end']]
            encoded = tokenizer(body, add_special_tokens=False, truncation=False,
                                return_offsets_mapping=True, verbose=False)
            offsets = encoded['offset_mapping']
            if not offsets:
                continue
            token_count += len(offsets)
            start, part = 0, 0
            while start < len(offsets):
                # 无提取元数据的短原始代码不附加虚构角色；真实注册/回调有显式边界
                prefix = f"Lua {block['role']} {block['name']} block {ordinal} part {part}:\n" if entry['blocks'] else ''
                budget = maximum - len(tokenizer.encode(prefix, add_special_tokens=True)) - 4
                if budget <= CODE_ENCODING_POLICY['overlap_tokens']:
                    raise ValueError('effect block label exceeds encoder context')
                end = min(start + budget, len(offsets))
                text = prefix + body[offsets[start][0]:offsets[end - 1][1]]
                while len(tokenizer.encode(text, add_special_tokens=True, truncation=False, verbose=False)) > maximum:
                    end -= 1
                    if end <= start:
                        raise ValueError('could not encode a complete Lua token')
                    text = prefix + body[offsets[start][0]:offsets[end - 1][1]]
                weight = end - start - (CODE_ENCODING_POLICY['overlap_tokens'] if start else 0)
                pieces.append((text, max(weight, 1)))
                if end == len(offsets):
                    break
                following = end - CODE_ENCODING_POLICY['overlap_tokens']
                if following <= start:
                    raise ValueError('code token window cannot make forward progress')
                start = following
                part += 1
        if not pieces:
            raise ValueError('effect source produced no encoder tokens')
        return pieces, token_count

    @staticmethod
    def _write_embedding_pair(output_path, embeddings, key_to_idx, metadata):
        """最后提交带文件摘要的生成清单，中断产生的混合资产会被校验拒绝"""
        output_path = Path(output_path).resolve()
        index_path = output_path.with_name(CODE_EMBEDDINGS_INDEX_FILENAME)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        embedding_temp = None
        index_temp = None
        metadata_temp = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='wb',
                prefix=f'.{output_path.name}.',
                suffix='.tmp',
                dir=output_path.parent,
                delete=False,
            ) as stream:
                embedding_temp = Path(stream.name)
                np.save(stream, embeddings, allow_pickle=False)
            with tempfile.NamedTemporaryFile(
                mode='w',
                prefix=f'.{index_path.name}.',
                suffix='.tmp',
                dir=index_path.parent,
                encoding='utf-8',
                delete=False,
            ) as stream:
                index_temp = Path(stream.name)
                json.dump(key_to_idx, stream, ensure_ascii=False)
            metadata['files'] = {'code_embeddings.npy': file_sha256(embedding_temp),
                                 CODE_EMBEDDINGS_INDEX_FILENAME: file_sha256(index_temp)}
            with tempfile.NamedTemporaryFile(mode='w', prefix='.code_embeddings_meta.', suffix='.tmp',
                                             dir=output_path.parent, encoding='utf-8', delete=False) as stream:
                metadata_temp = Path(stream.name)
                json.dump(metadata, stream, ensure_ascii=False, separators=(',', ':'))
            os.replace(embedding_temp, output_path)
            embedding_temp = None
            os.replace(index_temp, index_path)
            index_temp = None
            os.replace(metadata_temp, output_path.with_name(CODE_EMBEDDINGS_META_FILENAME))
            metadata_temp = None
        finally:
            for temporary in (embedding_temp, index_temp, metadata_temp):
                if temporary is not None and temporary.exists():
                    temporary.unlink()

    def generate_embeddings(
        self,
        kb_file='knowledge_base.json',
        output_file='code_embeddings.npy',
        incremental=True,
    ):
        """按内容和编码器身份复用或重生成，不接受未认证旧键和空源码"""
        if not os.path.exists(kb_file):
            print(f"[代码语义] 找不到 {kb_file}，请先执行 parse 构建知识库。")
            raise FileNotFoundError(kb_file)
            
        with open(kb_file, 'r', encoding='utf-8') as f:
            kb = json.load(f)
            
        entries = self._collect_effect_code(kb)
        if {key for key, _ in entries} != build_expected_code_semantic_keys(kb):
            raise ValueError('collected effect keys disagree with knowledge base')
        output_path = Path(output_file).resolve()
        if output_path.name != CODE_EMBEDDINGS_FILENAME or Path(kb_file).resolve().parent != output_path.parent:
            raise ValueError('knowledge base and canonical code_embeddings.npy must share one asset directory')
        existing = None
        if incremental:
            try:
                existing = validate_code_semantic_assets(output_path.parent)
            except (OSError, ValueError) as error:
                print(f"[代码语义] 现有资产校验失败，将安全全量重建: {error}")

        previous = existing.get('metadata') if existing else None
        reusable = bool(previous and previous['policy'] == CODE_ENCODING_POLICY
                        and previous['encoder']['name'] == self.model_name
                        and (getattr(self, 'revision', None) is None or self.revision == previous['encoder'].get('revision')))
        if reusable:
            self._pinned_revision = previous['encoder'].get('revision')
        needed = [(key, entry) for key, entry in entries if not reusable
                  or key not in previous['rows']
                  or previous['rows'][key]['encoding_sha256'] != entry['encoding_sha256']]
        # 已加载/本地自定义模型需要复核实际权重；远程模型按清单修订固定，未变时免加载
        identity = previous['encoder'] if reusable else None
        if needed or self.model is not None or Path(self.model_name).exists():
            identity = self._encoder_identity()
            if reusable and identity['fingerprint'] != previous['encoder']['fingerprint']:
                reusable = False
                needed = entries
        if any(entry['code'] is None for _, entry in needed):
            raise ValueError('unverified/changed vectors lack raw source; extract local Lua before regeneration')
        if not needed and existing and set(existing['index']) == {key for key, _ in entries}:
            validate_semantic_bundle(output_path.parent, knowledge_base_filename=Path(kb_file).name)
            print('[代码语义] 来源、编码器和分块规则均未变，已复用全部向量。')
            return {'reused': len(entries), 'generated': 0, 'chunks': 0}
        if identity is None:
            identity = self._encoder_identity()
        key_to_idx = {key: index for index, (key, _) in enumerate(entries)}
        embeddings = np.zeros((len(entries), identity['dimension']), dtype=np.float32)
        records = {}
        generated_keys = {key for key, _ in needed}
        if reusable:
            old_vectors = np.load(existing['embedding_path'], mmap_mode='r', allow_pickle=False)
            try:
                for key, entry in entries:
                    if key not in generated_keys:
                        embeddings[key_to_idx[key]] = old_vectors[existing['index'][key]]
                        records[key] = dict(previous['rows'][key], complete=entry['complete'])
            finally:
                old_vectors._mmap.close()
        print(f'[代码语义] 内容接续：复用 {len(entries) - len(needed)} 槽，生成/更新 {len(needed)} 槽。')
        total_chunks = 0
        # 有界小组分块，避免把全部长代码展开向量同时常驻内存
        for start in range(0, len(needed), 64):
            group = needed[start:start + 64]
            texts, owners, weights = [], [], []
            sums = np.zeros((len(group), identity['dimension']), dtype=np.float64)
            totals = np.zeros(len(group), dtype=np.float64)
            for owner, (key, entry) in enumerate(group):
                chunks, tokens = self._split_chunks(entry)
                records[key] = {'source_sha256': entry['source_sha256'], 'encoding_sha256': entry['encoding_sha256'],
                                'tokens': tokens, 'chunks': len(chunks), 'complete': entry['complete']}
                total_chunks += len(chunks)
                for text, weight in chunks:
                    texts.append(text)
                    owners.append(owner)
                    weights.append(weight)
            for offset in range(0, len(texts), 128):
                vectors = np.asarray(self._get_model().encode(texts[offset:offset + 128], batch_size=128, show_progress_bar=False), dtype=np.float32)
                if vectors.shape != (len(texts[offset:offset + 128]), identity['dimension']) or not np.isfinite(vectors).all():
                    raise ValueError('encoder returned invalid/nonfinite vectors')
                selection = np.asarray(owners[offset:offset + 128])
                factors = np.asarray(weights[offset:offset + 128], dtype=np.float64)
                np.add.at(sums, selection, vectors * factors[:, None])
                np.add.at(totals, selection, factors)
            sums /= totals[:, None]
            norms = np.linalg.norm(sums, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or (norms <= 1e-12).any():
                raise ValueError('code embedding collapsed to an invalid zero vector')
            sums /= norms
            for owner, (key, _) in enumerate(group):
                embeddings[key_to_idx[key]] = sums[owner]
            if start % 1024 == 0:
                print(f'[代码语义] 已完成 {min(start + 64, len(needed))}/{len(needed)} 槽；累计 {total_chunks} 个完整分块。', flush=True)
        metadata = {'format_version': CODE_EMBEDDINGS_META_VERSION, 'encoder': identity,
                    'policy': dict(CODE_ENCODING_POLICY), 'rows': records}
        self._write_embedding_pair(output_path, embeddings, key_to_idx, metadata)
        validate_semantic_bundle(
            output_path.parent,
            knowledge_base_filename=Path(kb_file).resolve().name,
        )
        invalidate_static_semantic_assets(output_path.parent)
        print(f"[代码语义] 提取完成，已保存至 {output_path} (维度: {embeddings.shape})")
        return {'reused': len(entries) - len(needed), 'generated': len(needed), 'chunks': total_chunks}

if __name__ == "__main__":
    embedder = CodeSemanticEmbedder()
    embedder.generate_embeddings()
