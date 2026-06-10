"""
问题理解模块

功能：
- 问题分类（基于规则 / TF-IDF）
- 关键词/实体提取（TF-IDF / TextRank）
- 为后续检索提供结构化查询表示

使用方式：
    qa = QuestionUnderstanding(cfg)
    result = qa.understand("什么是机器学习")
    # -> {"category": "ai_ml", "keywords": ["机器学习", "定义"], "query": "机器学习 定义"}
"""

import re
import math
from typing import List, Dict, Optional
from collections import Counter

from utils.tokenizer import ChineseTokenizer


class QuestionUnderstanding:
    """问题理解：分类 + 关键词提取"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.tokenizer = ChineseTokenizer(method="jieba")

        # 分类器
        self.classifier_method = self.config.get("classifier", "rule")
        self._categories = self.config.get("categories", [])

        # 关键词提取
        self.keyword_method = self.config.get("keyword_method", "tfidf")
        self.top_k = self.config.get("top_k_keywords", 5)

        # TF-IDF 关键词提取需要先 fit（在外部调用 fit_keywords_tfidf）
        self._tfidf_keyword_model = None

        # ── 规则分类词典 ──
        self._build_rule_classifier()

    # ══════════════════════════════════════════════════════════════════
    # 问题分类
    # ══════════════════════════════════════════════════════════════════

    def _build_rule_classifier(self):
        """构建基于关键词匹配的规则分类器"""
        self._rule_map = {
            "ai_ml": [
                "机器学习", "深度学习", "人工智能", "神经网络", "监督学习",
                "无监督学习", "强化学习", "过拟合", "欠拟合", "反向传播",
                "损失函数", "梯度下降", "激活函数", "注意力机制", "transformer",
                "bert", "gpt", "预训练", "微调", "模型蒸馏", "embedding",
                "迁移学习", "lora", "rag", "pca", "svm", "决策树", "随机森林",
                "xgboost", "聚类", "分类", "回归", "特征工程", "数据增强",
                "cnn", "rnn", "lstm", "resnet", "yolo", "数据归一化",
                "交叉验证", "batch normalization", "dropout", "softmax",
                "知识图谱", "情感分析", "文本分类", "命名实体识别", "分词",
                "目标检测", "图像分类", "图像分割", "计算机视觉", "nlp",
            ],
            "cloud_computing": [
                "云计算", "iaas", "paas", "saas", "docker", "kubernetes",
                "k8s", "微服务", "devops", "cd", "cdn", "容器化",
            ],
            "programming": [
                "编程", "git", "github", "api", "restful", "json", "xml",
                "面向对象", "oop", "函数式编程", "设计模式", "mvc",
                "单例模式", "工厂模式", "观察者模式", "代码审查", "代码重构",
                "grpc", "protobuf", "webassembly",
            ],
            "python": [
                "python", "列表推导式", "lambda", "装饰器", "生成器",
                "gil", "深拷贝", "浅拷贝", "__init__", "__new__",
                "上下文管理器", "垃圾回收", "property", "@staticmethod",
                "@classmethod", "django", "flask", "fastapi", "numpy",
                "pandas", "matplotlib", "pytorch", "tensor",
            ],
            "java": [
                "java", "jvm", "jre", "jdk", "final", "抽象类", "接口",
                "hashmap", "synchronized", "stream api", "volatile",
                "spring boot", "mybatis", "spring cloud",
            ],
            "javascript": [
                "javascript", "var", "let", "const", "闭包", "promise",
                "async", "await", "node.js", "dom", "typescript", "react",
                "vue.js", "vue",
            ],
            "c_cpp": [
                "c语言", "c++", "指针", "malloc", "calloc", "引用", "虚函数",
                "raii", "智能指针",
            ],
            "go": [
                "go语言", "goroutine", "channel", "defer", "interface",
            ],
            "rust": [
                "rust", "所有权", "生命周期",
            ],
            "database": [
                "sql", "mysql", "postgresql", "redis", "mongodb", "索引",
                "关系型数据库", "nosql", "join", "事务", "acid", "mvcc",
                "隔离级别", "脏读", "不可重复读", "幻读", "索引下推",
                "elasticsearch", "缓存雪崩", "缓存穿透",
            ],
            "os": [
                "操作系统", "进程", "线程", "死锁", "虚拟内存", "linux",
                "shell", "awk", "sed",
            ],
            "network": [
                "tcp/ip", "tcp", "udp", "http", "https", "dns", "负载均衡",
                "websocket", "三次握手", "四次挥手", "osi七层", "ipv4", "ipv6",
            ],
            "security": [
                "xss", "csrf", "https证书", "oauth", "jwt", "sql注入",
            ],
            "algorithms": [
                "栈", "队列", "链表", "堆", "二叉树", "哈希表", "图",
                "快速排序", "冒泡排序", "插入排序", "归并排序",
                "二分查找", "动态规划", "贪心算法", "布隆过滤器", "时间复杂度",
                "空间复杂度", "哈希冲突",
            ],
            "web": [
                "html", "css", "盒模型", "响应式设计", "跨域", "nginx",
                "graphql",
            ],
            "software_engineering": [
                "单元测试", "tdd", "敏捷开发", "ci/cd", "分布式系统",
                "cap定理", "消息队列", "kafka", "缓存", "速率限制",
            ],
            "big_data": [
                "大数据", "hadoop", "spark",
            ],
            "emerging_tech": [
                "量子计算", "ar", "vr", "物联网", "iot", "5g",
            ],
            "computer_basics": [
                "cpu", "gpu", "内存", "硬盘", "ssd", "hdd", "ascii",
            ],
            "blockchain": [
                "区块链", "比特币", "以太坊", "智能合约",
            ],
            "math": [
                "矩阵", "贝叶斯", "线性代数", "概率论",
            ],
        }

    def classify_by_rule(self, question: str) -> str:
        """基于规则的关键词匹配分类"""
        q_lower = question.lower()
        scores = {}
        for category, keywords in self._rule_map.items():
            score = sum(1 for kw in keywords if kw.lower() in q_lower)
            if score > 0:
                scores[category] = score

        if not scores:
            return "general"

        # 返回匹配关键词最多的分类
        return max(scores, key=scores.get)

    def classify(self, question: str) -> str:
        """统一分类入口"""
        if self.classifier_method == "rule":
            return self.classify_by_rule(question)
        else:
            return self.classify_by_rule(question)  # fallback

    # ══════════════════════════════════════════════════════════════════
    # 关键词提取
    # ══════════════════════════════════════════════════════════════════

    def extract_keywords_tfidf(self, question: str) -> List[str]:
        """基于 TF-IDF 的关键词提取（需先调用 fit_keywords_tfidf）"""
        if self._tfidf_keyword_model is None:
            raise ValueError("TF-IDF 模型未训练，请先调用 fit_keywords_tfidf()")

        tokens = self.tokenizer.tokenize(question)
        tokens = [t for t in tokens if len(t) > 1]  # 过滤单字
        if not tokens:
            return []

        scores = {}
        for token in set(tokens):
            # TF: 词在当前文档中频率
            tf = tokens.count(token) / len(tokens)
            # IDF: 从训练好的模型中获取
            idf = self._tfidf_keyword_model.get(token, 1.0)
            scores[token] = tf * idf

        sorted_keywords = sorted(scores.items(), key=lambda x: -x[1])
        return [kw for kw, _ in sorted_keywords[:self.top_k]]

    def extract_keywords_textrank(self, question: str) -> List[str]:
        """基于 TextRank 的关键词提取（简易实现）"""
        tokens = self.tokenizer.tokenize(question)
        tokens = [t for t in tokens if len(t) > 1]
        if len(tokens) < 2:
            return tokens

        # 简易窗口共现
        window_size = 3
        scores = {t: 1.0 for t in set(tokens)}
        for i in range(len(tokens)):
            for j in range(i + 1, min(i + window_size, len(tokens))):
                if tokens[i] != tokens[j]:
                    scores[tokens[i]] = scores.get(tokens[i], 1.0) + 1.0 / (j - i)

        sorted_kw = sorted(scores.items(), key=lambda x: -x[1])
        return [kw for kw, _ in sorted_kw[:self.top_k]]

    def extract_keywords(self, question: str) -> List[str]:
        """统一关键词提取入口"""
        if self.keyword_method == "tfidf" and self._tfidf_keyword_model:
            return self.extract_keywords_tfidf(question)
        elif self.keyword_method == "textrank":
            return self.extract_keywords_textrank(question)
        else:
            # 默认：提取长度 > 1 的名词/关键词
            tokens = self.tokenizer.tokenize(question)
            return [t for t in tokens if len(t) > 1][:self.top_k]

    def fit_keywords_tfidf(self, corpus: List[str]):
        """
        在语料上训练 TF-IDF 模型，用于关键词提取。

        Args:
            corpus: 文档列表（所有 FAQ 问题 + 答案文本）
        """
        from collections import Counter
        import math

        # 分词
        tokenized = []
        for doc in corpus:
            tokens = self.tokenizer.tokenize(doc)
            tokenized.append([t for t in tokens if len(t) > 1])

        doc_count = len(tokenized)
        idf = {}

        for doc_tokens in tokenized:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                idf[token] = idf.get(token, 0) + 1

        # 计算 IDF
        for token in idf:
            idf[token] = math.log((doc_count + 1) / (idf[token] + 1)) + 1

        self._tfidf_keyword_model = idf

    # ══════════════════════════════════════════════════════════════════
    # 主流程
    # ══════════════════════════════════════════════════════════════════

    def understand(self, question: str) -> Dict:
        """
        完整的问题理解流程。

        Returns:
            {
                "category": str,      # 问题分类
                "keywords": [str],    # 关键词列表
                "query": str,         # 扩展后的查询字符串
                "original": str,      # 原始问题
            }
        """
        category = self.classify(question)
        keywords = self.extract_keywords(question)

        # 构造扩展查询：原始问题 + 关键词
        query = f"{question} {' '.join(keywords)}"

        return {
            "category": category,
            "keywords": keywords,
            "query": query,
            "original": question,
        }
