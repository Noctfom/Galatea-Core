# online_fetcher.py
import os
import requests
import time
import json

class BaseFetcher:
    def __init__(self):
        self.name = "Base"
    def test_connection(self): raise NotImplementedError
    def fetch_decks(self, limit, target_dir, **kwargs): raise NotImplementedError

class YGOProDeckFetcher(BaseFetcher):
    def __init__(self):
        super().__init__()
        self.name = "YGOProDeck"
        self.base_url = "https://ygoprodeck.com/api/decks/getDecks.php"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 GalateaAI/1.0',
            'Accept': 'application/json'
        }
        self.last_request_time = 0

    def _rate_limit(self):
        """核心防封禁节流阀"""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < 0.15: time.sleep(0.15 - elapsed)
        self.last_request_time = time.time()

    def test_connection(self):
        self._rate_limit()
        try:
            resp = requests.get(f"{self.base_url}?limit=1&offset=0", headers=self.headers, timeout=10)
            if resp.status_code == 200: return True, "✅ YGOProDeck 握手成功！"
            else: return False, f"❌ 连接失败: HTTP 状态码 {resp.status_code}"
        except requests.RequestException as e:
            return False, f"❌ 网络异常: {str(e)}"

    def fetch_decks(self, limit=30, target_dir="./decks/ygoprodeck_meta", **kwargs):
        os.makedirs(target_dir, exist_ok=True)
        
        api_category = kwargs.get('api_category', 'All')
        # 强制将 offset 对齐到 20 的整数倍，迎合 WordPress 分页机制
        base_offset = (kwargs.get('offset', 0) // 20) * 20
        
        success_count = 0
        current_offset = base_offset
        
        # 🌟 核心突破：自动循环翻页机制
        while success_count < limit:
            # 每次最多只讨要 20 个，遵循服务器硬性底线
            fetch_amount = min(20, limit - success_count)
            params = {"offset": current_offset, "limit": fetch_amount} 
            if api_category != "All": params["_sft_category"] = api_category
                
            self._rate_limit()
            try:
                resp = requests.get(self.base_url, params=params, headers=self.headers, timeout=15)
                if resp.status_code != 200: break
                    
                data = resp.json()
                decks = data if isinstance(data, list) else data.get('decks', [])
                
                if not decks:
                    # 如果一开始就踩空了，说明该分类总数没那么大
                    if success_count == 0 and current_offset > 0:
                        # 偏移量减半，并向下取整对齐 20
                        next_offset = current_offset // 2
                        current_offset = (next_offset // 20) * 20
                        time.sleep(0.5) # 给服务器一点喘息时间
                        continue # 重新发起请求
                    else:
                        # 如果是已经抓了一部分后空了，或者 offset 已经是 0 依然为空，说明真到底了
                        break
                
                for d in decks:
                    if success_count >= limit: break
                    
                    deck_name = d.get('deck_name', f"Deck_{int(time.time())}")
                    safe_name = "".join([c for c in deck_name if c.isalpha() or c.isdigit() or c==' ' or c=='-' or c=='_']).strip()
                    if not safe_name: safe_name = f"Deck_{int(time.time())}"
                    
                    try:
                        main_deck = json.loads(d.get('main_deck', '[]'))
                        extra_deck = json.loads(d.get('extra_deck', '[]'))
                        side_deck = json.loads(d.get('side_deck', '[]'))
                    except json.JSONDecodeError: continue 
                    
                    ydk_content = f"#created by Galatea YGOProDeck Fetcher\n#main\n" + "\n".join([str(c) for c in main_deck]) + "\n"
                    ydk_content += "#extra\n" + "\n".join([str(c) for c in extra_deck]) + "\n"
                    ydk_content += "!side\n" + "\n".join([str(c) for c in side_deck]) + "\n"
                    
                    # 🌟 加盟了 deckNum 后缀，防止同名卡组互相覆盖
                    unique_id = d.get('deckNum', int(time.time()))
                    file_path = os.path.join(target_dir, f"{safe_name}_{unique_id}.ydk")
                    
                    with open(file_path, 'w', encoding='utf-8') as f: f.write(ydk_content)
                    success_count += 1
                    
                # 翻到下一页
                current_offset += 20
                
            except Exception as e:
                return False, f"❌ 抓取在第 {success_count} 个时中断: {str(e)}"
                
        if success_count == 0:
            return False, "❌ 未能抓取到任何卡组，可能是参数错误或该分类下没有数据。"
            
        return True, f"成功抓取并生成了 {success_count} 个卡组！"