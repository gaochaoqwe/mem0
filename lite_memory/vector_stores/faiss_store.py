"""
FAISS向量存储实现 - 支持离线部署
重构自原项目，适配Python 3.8，简化功能
"""

import logging
import os
import pickle
import uuid
import sys
import subprocess
from typing import List, Dict, Optional, Any
from pathlib import Path

import numpy as np

from lite_memory.vector_stores.base import VectorStoreBase, SearchResult
from lite_memory.config import VectorStoreConfig

# 设置日志
logger = logging.getLogger(__name__)

# 尝试导入faiss
try:
    import faiss
    # 抑制faiss的警告日志
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("faiss.loader").setLevel(logging.WARNING)
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS库未安装")


class FAISSStore(VectorStoreBase):
    """FAISS向量存储实现"""
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        初始化FAISS向量存储
        
        Args:
            config: 向量存储配置
        """
        super().__init__(config)
        
        if config is None:
            config = VectorStoreConfig()
        self.config = config
        
        # 检查并安装FAISS
        self._ensure_faiss_available()
        
        # 初始化存储结构
        self.index = None
        self.docstore = {}  # id -> metadata
        self.index_to_id = {}  # faiss_index -> id
        
        # 确保目录存在
        os.makedirs(self.config.path, exist_ok=True)
        
        # 尝试加载现有索引
        self._load_index()
    
    def _ensure_faiss_available(self):
        """确保FAISS库可用"""
        if not FAISS_AVAILABLE:
            self._install_faiss()
    
    def _install_faiss(self):
        """安装FAISS库"""
        try:
            print("正在安装FAISS库...")
            # 优先安装CPU版本，兼容性更好
            subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
            print("FAISS安装成功")
            
            # 重新导入
            global faiss, FAISS_AVAILABLE
            import faiss
            FAISS_AVAILABLE = True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FAISS安装失败: {e}")
            raise RuntimeError(
                "无法安装FAISS库。请手动安装：\n"
                "pip install faiss-cpu  # CPU版本\n"
                "或\n"
                "pip install faiss-gpu  # GPU版本（需要CUDA支持）"
            )
    
    def _get_index_path(self) -> str:
        """获取索引文件路径"""
        return os.path.join(self.config.path, f"{self.config.collection_name}.faiss")
    
    def _get_docstore_path(self) -> str:
        """获取文档存储路径"""
        return os.path.join(self.config.path, f"{self.config.collection_name}.pkl")
    
    def _load_index(self):
        """加载现有索引"""
        index_path = self._get_index_path()
        docstore_path = self._get_docstore_path()
        
        if os.path.exists(index_path) and os.path.exists(docstore_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(docstore_path, "rb") as f:
                    self.docstore, self.index_to_id = pickle.load(f)
                logger.info(f"加载索引成功，包含 {self.index.ntotal} 个向量")
            except Exception as e:
                logger.warning(f"加载索引失败: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """创建新索引"""
        # 根据距离策略选择索引类型
        if self.config.distance_strategy.lower() in ["cosine", "inner_product"]:
            self.index = faiss.IndexFlatIP(self.config.embedding_dims)
        else:
            self.index = faiss.IndexFlatL2(self.config.embedding_dims)
        
        self.docstore = {}
        self.index_to_id = {}
        logger.info(f"创建新索引，维度: {self.config.embedding_dims}")
    
    def _save_index(self):
        """保存索引到磁盘"""
        if not self.index:
            return
        
        try:
            index_path = self._get_index_path()
            docstore_path = self._get_docstore_path()
            
            faiss.write_index(self.index, index_path)
            with open(docstore_path, "wb") as f:
                pickle.dump((self.docstore, self.index_to_id), f)
                
            logger.debug("索引保存成功")
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
    
    def _apply_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """应用过滤条件"""
        if not filters or not metadata:
            return True
        
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        
        return True
    
    def create_collection(self, name: str) -> bool:
        """
        创建集合
        
        Args:
            name: 集合名称
            
        Returns:
            bool: 是否创建成功
        """
        try:
            self.config.collection_name = name
            self._create_new_index()
            self._save_index()
            return True
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def insert(self, vectors: List[List[float]], ids: List[str],
               metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        插入向量
        
        Args:
            vectors: 向量列表
            ids: ID列表
            metadata: 元数据列表
            
        Returns:
            bool: 是否插入成功
        """
        if not self.index:
            logger.error("索引未初始化")
            return False
        
        if len(vectors) != len(ids):
            logger.error("向量数量与ID数量不匹配")
            return False
        
        if metadata is None:
            metadata = [{} for _ in range(len(vectors))]
        elif len(metadata) != len(vectors):
            logger.error("元数据数量与向量数量不匹配")
            return False
        
        try:
            # 转换为numpy数组
            vectors_np = np.array(vectors, dtype=np.float32)
            
            # 归一化处理
            if self.config.normalize_L2 and self.config.distance_strategy.lower() == "cosine":
                faiss.normalize_L2(vectors_np)
            
            # 添加到索引
            starting_idx = len(self.index_to_id)
            self.index.add(vectors_np)
            
            # 更新文档存储
            for i, (vector_id, meta) in enumerate(zip(ids, metadata)):
                self.docstore[vector_id] = meta.copy()
                self.index_to_id[starting_idx + i] = vector_id
            
            # 保存索引
            if self.config.auto_save:
                self._save_index()
            
            logger.info(f"成功插入 {len(vectors)} 个向量")
            return True
            
        except Exception as e:
            logger.error(f"插入向量失败: {e}")
            return False
    
    def search(self, vector: List[float], limit: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        搜索相似向量
        
        Args:
            vector: 查询向量
            limit: 返回结果数量
            filters: 过滤条件
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        if not self.index:
            logger.error("索引未初始化")
            return []
        
        try:
            # 转换为numpy数组
            query_vector = np.array([vector], dtype=np.float32)
            
            # 归一化处理
            if self.config.normalize_L2 and self.config.distance_strategy.lower() == "cosine":
                faiss.normalize_L2(query_vector)
            
            # 搜索时获取更多结果用于过滤
            fetch_k = limit * 2 if filters else limit
            scores, indices = self.index.search(query_vector, min(fetch_k, self.index.ntotal))
            
            results = []
            for i in range(len(indices[0])):
                faiss_idx = indices[0][i]
                score = float(scores[0][i])
                
                # 跳过无效索引
                if faiss_idx == -1:
                    continue
                
                # 获取ID和元数据
                vector_id = self.index_to_id.get(faiss_idx)
                if vector_id is None:
                    continue
                
                metadata = self.docstore.get(vector_id, {})
                
                # 应用过滤条件
                if filters and not self._apply_filters(metadata, filters):
                    continue
                
                results.append(SearchResult(
                    id=vector_id,
                    score=score,
                    metadata=metadata.copy()
                ))
                
                # 达到限制数量就停止
                if len(results) >= limit:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def delete(self, id: str) -> bool:
        """
        删除向量（标记删除，不重建索引）
        
        Args:
            id: 向量ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
            if id not in self.docstore:
                logger.warning(f"向量 {id} 不存在")
                return False
            
            # 从文档存储中删除
            del self.docstore[id]
            
            # 从ID映射中删除
            faiss_idx_to_remove = None
            for faiss_idx, vector_id in self.index_to_id.items():
                if vector_id == id:
                    faiss_idx_to_remove = faiss_idx
                    break
            
            if faiss_idx_to_remove is not None:
                del self.index_to_id[faiss_idx_to_remove]
            
            # 保存索引
            if self.config.auto_save:
                self._save_index()
            
            logger.info(f"删除向量 {id} 成功")
            return True
            
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            return False
    
    def update(self, id: str, vector: Optional[List[float]] = None,
               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        更新向量（简化实现：删除后重新插入）
        
        Args:
            id: 向量ID
            vector: 新向量
            metadata: 新元数据
            
        Returns:
            bool: 是否更新成功
        """
        try:
            if id not in self.docstore:
                logger.warning(f"向量 {id} 不存在")
                return False
            
            # 获取当前元数据
            current_metadata = self.docstore[id].copy()
            
            # 更新元数据
            if metadata is not None:
                current_metadata.update(metadata)
            
            # 如果只更新元数据
            if vector is None:
                self.docstore[id] = current_metadata
                if self.config.auto_save:
                    self._save_index()
                return True
            
            # 如果更新向量，需要删除后重新插入
            self.delete(id)
            return self.insert([vector], [id], [current_metadata])
            
        except Exception as e:
            logger.error(f"更新向量失败: {e}")
            return False
    
    def get(self, id: str) -> Optional[SearchResult]:
        """
        获取向量
        
        Args:
            id: 向量ID
            
        Returns:
            Optional[SearchResult]: 向量数据，不存在时返回None
        """
        if id not in self.docstore:
            return None
        
        return SearchResult(
            id=id,
            score=0.0,  # 直接获取不计算分数
            metadata=self.docstore[id].copy()
        )
    
    def list_all(self, filters: Optional[Dict[str, Any]] = None,
                 limit: int = 100) -> List[SearchResult]:
        """
        列出所有向量
        
        Args:
            filters: 过滤条件
            limit: 返回结果数量
            
        Returns:
            List[SearchResult]: 向量列表
        """
        results = []
        count = 0
        
        for vector_id, metadata in self.docstore.items():
            if filters and not self._apply_filters(metadata, filters):
                continue
            
            results.append(SearchResult(
                id=vector_id,
                score=0.0,
                metadata=metadata.copy()
            ))
            
            count += 1
            if count >= limit:
                break
        
        return results
    
    def count(self) -> int:
        """
        获取向量总数
        
        Returns:
            int: 向量总数
        """
        return len(self.docstore)
    
    def clear(self) -> bool:
        """
        清空所有向量
        
        Returns:
            bool: 是否清空成功
        """
        try:
            self._create_new_index()
            self._save_index()
            logger.info("清空向量存储成功")
            return True
        except Exception as e:
            logger.error(f"清空向量存储失败: {e}")
            return False
    
    def save(self):
        """手动保存索引"""
        self._save_index()
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取存储信息
        
        Returns:
            Dict[str, Any]: 存储信息
        """
        return {
            "collection_name": self.config.collection_name,
            "count": self.count(),
            "dimension": self.config.embedding_dims,
            "distance_strategy": self.config.distance_strategy,
            "path": self.config.path
        }