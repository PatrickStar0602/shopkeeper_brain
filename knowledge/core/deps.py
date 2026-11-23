from functools import lru_cache
from knowledge.services.upload_service import UploadService
from knowledge.services.query_service import QueryService


# @cache缓存注解
@lru_cache  #淘汰策略：最近最少使用
def get_upload_service():
    return UploadService()

@lru_cache
def get_query_service():
    return QueryService()

