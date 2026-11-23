import logging
import os.path
import shutil
import time
import uuid
from datetime import datetime

from fastapi import UploadFile
from langgraph.graph import StateGraph

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import FileProcessingError
from knowledge.processor.import_process.main_graph import create_import_graph
from knowledge.processor.import_process.state import get_default_state
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_util import add_running_task, add_done_task, add_node_duration
from knowledge.utils.task_util import get_task_info, update_task_status


class UploadService:
    def process_upload_file(self, file: UploadFile):
        #生成当前文件上传任务的task_id
        task_id = uuid.uuid4().hex[:8]

        update_task_status(task_id, "running")
        add_running_task(task_id, "upload_file")
        start_time = time.time()

        #获取文件在服务器进行保存的目录路径
        base_dir = get_local_base_dir()
        file_dir = os.path.join(base_dir, task_id)

        #将文件保存到file_dir
        import_file_path = self.save_upload_file_to_local(file, file_dir)
        print(import_file_path)

        #将用户上传的文件保存到MinIO
        remote_url = self.save_upload_file_to_minio(import_file_path, file.filename)

        #mysql储存文件及路径信息

        #启动文件导入状态图
        #self.run_import_graph(import_file_path, file_dir)

        #返回结果
        end_time = time.time()
        add_done_task(task_id, "upload_file")
        add_node_duration(task_id, "upload_file", end_time - start_time)
        message = "文件上传成功，正在处理"
        return message, task_id, import_file_path, file_dir


    def save_upload_file_to_local(self, file, file_dir)->str:
        #创建目录
        os.makedirs(file_dir, exist_ok=True)
        #生成文件路径
        import_file_path = os.path.join(file_dir, file.filename)
        try:
            with open(import_file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        except IOError as e:
            raise FileProcessingError(f"文件保存失败: {e}")
        return import_file_path


    def save_upload_file_to_minio(self, import_file_path, filename)->str:
        #获取MinIO客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            logging.Logger.info(f"获取MinIO客户端失败: {e}")
            return
        #上传文件
        bucket_name = get_config().minio_bucket
        object_name = f"origin_files/{datetime.now().strftime('%Y%m%d')}/{filename}"
        try:
            minio_client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=import_file_path
            )
        except Exception as e:
            logging.Logger.error(f"文件上传失败: {e}")
            return
        #返回文件在MinIO的远程url
        remote_url = f"{get_config().get_minio_base_url()}/{bucket_name}/{object_name}"

        return remote_url

    def run_import_graph(self, import_file_path, file_dir, task_id):
        setup_logging(logging.DEBUG)
        graph: StateGraph = create_import_graph()
        state = get_default_state()
        state["import_file_path"] = import_file_path
        state["file_dir"] = file_dir
        state["task_id"] = task_id

        # new_state = graph.invoke(state)
        try:
            for event in graph.stream(state):
                for node, output in event.items():
                    logging.info(f"{node}节点处理完成, 输出结果为: {output}")
            update_task_status(task_id, "completed")
        except Exception as e:
            update_task_status(task_id, "failed")

    def get_status(self, task_id:str):
        return get_task_info(task_id)
