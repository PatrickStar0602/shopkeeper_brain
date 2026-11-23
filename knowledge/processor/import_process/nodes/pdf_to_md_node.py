import subprocess
import sys
import time

from pathlib import Path
from typing import Tuple
from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.exceptions import StateFieldError, PdfConversionError
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):

    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        #校验state参数
        pdf_path_obj, file_dir_obj = self._validate_state(state)
        #调用MinerU工具解析PDF
        process_code = self._execute_mineru(pdf_path_obj, file_dir_obj)
        #获取转换生产的MD文件路径
        md_path = self._get_md_path(pdf_path_obj, file_dir_obj)
        #更新state
        state["md_path"] = md_path
        #返回更新后的state
        return state


    def _validate_state(self, state: ImportGraphState) -> Tuple[Path, Path]:
        self.log_step("Step1", "校验state参数")
        pdf_path = state.get("pdf_path")
        file_dir = state.get("file_dir")

        if not pdf_path:
            raise StateFieldError(
                node_name=self.name,
                field_name="pdf_path",
                message="pdf_path 为空，上游节点可能未返回 state"
            )

        if not file_dir:
            raise StateFieldError(
                node_name=self.name,
                field_name="file_dir",
                message="file_dir 为空，上游节点可能未返回 state"
            )

        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise StateFieldError(
                node_name = self.name,
                field_name = "pdf_path",
                message = "PDF文件不存在"
            )

        file_dir_obj = Path(file_dir)
        if not file_dir_obj.exists():
            raise StateFieldError(
                node_name=self.name,
                field_name="file_dir",
                message="文件输出目录不存在"
            )

        return pdf_path_obj, file_dir_obj

    def _execute_mineru(self, pdf_path_obj:Path, file_dir_obj:Path)->int:
        self.log_step("Step2", "调用MinerU工具解析PDF")
        #定位当前虚拟环境内的 mineru 可执行文件，不依赖 PATH
        mineru_exe = Path(sys.executable).with_name("mineru.exe")
        if not mineru_exe.exists():
            raise PdfConversionError(
                node_name=self.name,
                message=f"未找到 mineru 可执行文件：{mineru_exe}"
            )

        #构建cmd指令
        cmd = [
            str(mineru_exe),
            "-p",
            str(pdf_path_obj),
            "-o",
            str(file_dir_obj),
            "-b",
            "pipeline",
            "--source",
            "local",
        ]

        #执行指令
        start_time = time.time()
        process = subprocess.Popen(
            args=cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        )

        #获取命令的日志
        for line in process.stdout:
            self.logger.info(f"MinerU:{line.strip()}")
        process_code = process.wait()
        end_time = time.time()
        if process_code != 0:
            raise PdfConversionError(
                node_name=self.name,
                message="MinerU转换失败"
            )
        else:
            self.logger.info(f"MinerU转换成功，耗时{end_time - start_time}秒")
        return process_code

    def _get_md_path(self, pdf_path_obj:Path, file_dir_obj:Path)->str:
        self.log_step("Step3", "获取转换生产的MD文件路径")
        #提取PDF文件名
        pdf_stem = pdf_path_obj.stem
        #拼接MD文件路径
        md_path = file_dir_obj / pdf_stem / "auto" / f"{pdf_stem}.md"
        if not md_path.exists():
            raise PdfConversionError(
                node_name=self.name,
                message="转换生产的MD文件不存在"
            )
        #返回
        return str(md_path)

#测试
if __name__ == "__main__":
    import json

    state = {
        "task_id": "",
        "is_pdf_read_enabled": True,
        "is_md_read_enabled": False,
        "file_dir": "D:\\Python\\shopkeeper_brain\\knowledge\\processor\\import_process\\import_files",
        "import_file_path": "D:\\Python\\shopkeeper_brain\\knowledge\\processor\\import_process\\import_files\\万用表RS-12的使用.pdf",
        "pdf_path": "D:\\Python\\shopkeeper_brain\\knowledge\\processor\\import_process\\import_files\\万用表RS-12的使用.pdf",
        "md_path": "",
        "file_title": "万用表RS-12的使用",
        "md_content": "",
        "chunks": [],
        "item_name": ""
    }

    node = PdfToMdNode()
    node(state)

    json_str = json.dumps(state, indent=4, ensure_ascii=False)
    print(json_str)