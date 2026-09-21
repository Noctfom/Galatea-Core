# -*- coding: utf-8 -*-
# 本文件提供旧编码器兼容的语义知识库接口，实际数据统一来自模型侧静态查表。

from pathlib import Path

from card_vocab import get_default_card_vocabulary
from effect_slot_binding import register_runtime_effect_binding_catalog
from protocol_v3_audit import register_compiled_semantic_audit_catalog
from semantic_lookup import get_static_semantic_lookup


class SemanticKnowledgeBase:
    """复用统一静态查表，避免 Encoder 与网络重复解析和保存语义资产"""

    def __init__(self, kb_path="knowledge_base.json", card_vocabulary=None):
        """严格加载完整语义查表并登记运行时效果槽审计目录"""
        self.card_vocabulary = card_vocabulary or get_default_card_vocabulary()
        knowledge_base_path = Path(kb_path).resolve()
        self.knowledge_base_path = knowledge_base_path
        self.lookup = get_static_semantic_lookup(
            knowledge_base_path.parent,
            card_vocabulary=self.card_vocabulary,
            knowledge_base_filename=knowledge_base_path.name,
        )
        self.cat2idx = self.lookup.category_to_index
        self.req2idx = self.lookup.requirement_to_index
        self.num_cats = len(self.cat2idx)
        self.req_dim = 128
        self.code_dim = int(self.lookup.code_dictionary.shape[1])
        self.code_embeddings = self.lookup.code_dictionary[1:]
        register_runtime_effect_binding_catalog(
            self.lookup.runtime_effect_bindings
        )
        register_compiled_semantic_audit_catalog(
            self.lookup.semantic_card_slots,
            self.lookup.runtime_effect_bindings,
        )

    def get_card_semantics(self, card_id):
        """按真实卡密返回与既有 Encoder 完全同形的静态语义视图"""
        return self.lookup.get_card_semantics(card_id)

    def is_current(self):
        """判断在线同步后的语义资产是否仍对应当前缓存对象"""
        current = get_static_semantic_lookup(
            self.knowledge_base_path.parent,
            card_vocabulary=self.card_vocabulary,
            knowledge_base_filename=self.knowledge_base_path.name,
        )
        return current is self.lookup
