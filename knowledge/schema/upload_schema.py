from pydantic import BaseModel, Field
from typing import List, Dict


class UploadResponse(BaseModel):
    message:str = Field(..., description="提示信息")
    task_id:str = Field(..., description="任务ID")

class TaskStatusResponse(BaseModel):
    status: str = Field(..., description="任务状态")
    done_list:List[str] = Field(..., description="已完成的文件列表")
    running_list:List[str] = Field(..., description="正在处理的文件列表")
    durations:Dict[str,float] = Field(..., description="文件处理耗时")
