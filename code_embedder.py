import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import os

class CodeSemanticEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"⌛ 正在加载预训练代码理解模型 {model_name}...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        
    def generate_embeddings(self, kb_file='knowledge_base.json', output_file='code_embeddings.npy'):
        if not os.path.exists(kb_file):
            print(f"❌ 找不到 {kb_file}，请先执行 parse 构建知识库。")
            return
            
        with open(kb_file, 'r', encoding='utf-8') as f:
            kb = json.load(f)
            
        keys = []
        codes = []
        
        # 遍历所有卡片的所有 8 个槽位
        for card_id, data in kb.items():
            for eff in data.get('effects', []):
                slot_idx = eff.get('slot', 1) - 1 # 转为 0-7 索引
                if slot_idx >= 8: continue
                
                code_text = eff.get('raw_code', '')
                keys.append(f"{card_id}_{slot_idx}")
                codes.append(code_text)
        
        print(f"⚙️ 正在为 {len(codes)} 个效果槽位提取全量高维特征...")
        
        # 批量编码 (空字符串会生成中性向量，安全无害)
        embeddings = self.model.encode(codes, batch_size=128, show_progress_bar=True)
        key_to_idx = {k: i for i, k in enumerate(keys)}
        
        np.save(output_file, embeddings)
        with open('code_embeddings_idx.json', 'w', encoding='utf-8') as f:
            json.dump(key_to_idx, f)
            
        print(f"✅ 全量代码语义提取完成，已保存至 {output_file} (维度: {embeddings.shape})")

if __name__ == "__main__":
    embedder = CodeSemanticEmbedder()
    embedder.generate_embeddings()