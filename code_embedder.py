import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import os
import tempfile
from pathlib import Path

from semantic_assets import (
    build_expected_code_semantic_keys,
    CODE_EMBEDDINGS_INDEX_FILENAME,
    validate_code_semantic_assets,
    validate_semantic_bundle,
)

class CodeSemanticEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None

    def _get_model(self):
        """仅在确实需要计算新向量时加载代码语义模型"""
        if self.model is None:
            print(f"[代码语义] 正在加载预训练代码理解模型 {self.model_name}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
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
                entries.append((
                    f"{card_id}_{slot_idx}",
                    effect.get('raw_code', ''),
                ))
        return entries

    @staticmethod
    def _write_embedding_pair(output_path, embeddings, key_to_idx):
        """先写临时文件，再替换代码向量与索引，避免中断留下半文件"""
        output_path = Path(output_path).resolve()
        index_path = output_path.with_name(CODE_EMBEDDINGS_INDEX_FILENAME)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        embedding_temp = None
        index_temp = None
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
            os.replace(embedding_temp, output_path)
            embedding_temp = None
            os.replace(index_temp, index_path)
            index_temp = None
        finally:
            for temporary in (embedding_temp, index_temp):
                if temporary is not None and temporary.exists():
                    temporary.unlink()

    def generate_embeddings(
        self,
        kb_file='knowledge_base.json',
        output_file='code_embeddings.npy',
        incremental=True,
    ):
        """生成代码语义向量，并在资产一致时仅追加新增效果槽"""
        if not os.path.exists(kb_file):
            print(f"[代码语义] 找不到 {kb_file}，请先执行 parse 构建知识库。")
            return
            
        with open(kb_file, 'r', encoding='utf-8') as f:
            kb = json.load(f)
            
        entries = self._collect_effect_code(kb)
        current_keys = build_expected_code_semantic_keys(kb)
        output_path = Path(output_file).resolve()
        existing = None
        if incremental:
            try:
                existing = validate_code_semantic_assets(output_path.parent)
            except (OSError, ValueError) as error:
                print(f"[代码语义] 现有资产校验失败，将安全全量重建: {error}")

        if existing is not None:
            key_to_idx = dict(existing['index'])
            stale_keys = set(key_to_idx).difference(current_keys)
            expected_dimension = self._get_expected_dimension()
            if stale_keys or existing['shape'][1] != expected_dimension:
                print("[代码语义] 现有资产与当前知识库或向量模型不一致，将安全全量重建。")
                existing = None

        if existing is None:
            keys = [key for key, _ in entries]
            codes = [code for _, code in entries]
            print(f"[代码语义] 正在为 {len(codes)} 个效果槽位提取全量高维特征...")
            if codes:
                model = self._get_model()
                embeddings = np.asarray(
                    model.encode(codes, batch_size=128, show_progress_bar=True),
                    dtype=np.float32,
                )
            else:
                embeddings = np.zeros(
                    (0, self._get_expected_dimension()),
                    dtype=np.float32,
                )
            key_to_idx = {key: index for index, key in enumerate(keys)}
        else:
            key_to_idx = dict(existing['index'])
            new_entries = [entry for entry in entries if entry[0] not in key_to_idx]
            if new_entries:
                print(f"[代码语义] 检测到 {len(new_entries)} 个新增效果槽，仅提取增量代码语义...")
                model = self._get_model()
                embeddings = np.load(
                    existing['embedding_path'],
                    allow_pickle=False,
                )
                new_vectors = np.asarray(
                    model.encode(
                        [code for _, code in new_entries],
                        batch_size=128,
                        show_progress_bar=True,
                    ),
                    dtype=embeddings.dtype,
                )
                start_index = embeddings.shape[0]
                embeddings = np.concatenate((embeddings, new_vectors), axis=0)
                for offset, (key, _) in enumerate(new_entries):
                    key_to_idx[key] = start_index + offset
            else:
                print("[代码语义] 向量已覆盖当前知识库，无需重复提取。")
                return

        self._write_embedding_pair(output_path, embeddings, key_to_idx)
        validate_semantic_bundle(
            output_path.parent,
            knowledge_base_filename=Path(kb_file).resolve().name,
        )
        print(f"[代码语义] 提取完成，已保存至 {output_path} (维度: {embeddings.shape})")

if __name__ == "__main__":
    embedder = CodeSemanticEmbedder()
    embedder.generate_embeddings()
