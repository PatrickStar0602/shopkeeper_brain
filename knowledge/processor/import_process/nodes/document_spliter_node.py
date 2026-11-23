import json
import logging
import os
import re
from typing import Tuple, List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.exceptions import DocumentSplitError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.markdown_utils import MarkdownTableLinearizer


class DocumentSpliterNode(BaseNode):

    name = "document_spliter_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 获取文档内容和标题
        md_content, file_title = self._get_inputs(state)
        #按照md文档的标题切分文档
        section_list = self._split_by_headings(md_content, file_title)
        #对按照标题切分后的块进行二次切分与合并处理
        final_section_list = self._split_and_merge_sections(section_list)
        #对切分后的结果进行组装
        chunks = self._assmble_chunks(final_section_list)
        state["chunks"] = chunks

        self._log_summary(md_content, chunks)
        #备份chunks
        self._backup_chunks(state, chunks)

        return state

    def _get_inputs(self, state: ImportGraphState) -> Tuple[str, str]:
        self.log_step("获取文档内容和标题")

        # 获取并验证md_content
        md_content = state.get("md_content")
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        else:
            raise DocumentSplitError(
                node_name = self.name,
                message = "未找到md文档数据"
            )
        #获取文件标题
        file_title = state.get("file_title", "")
        return md_content, file_title

    def _split_by_headings(self, md_content, file_title)-> List[Dict[str, Any]]:
        self.log_step( "按照md文档的标题切分文档")

        is_in_code_block = False
        section_list = []
        current_section = []
        current_title = ""
        current_title_level = 0
        hierarchy = [""]*7

        def _get_section():
            body_content = "\n".join(current_section)
            if body_content:
                title = current_title if current_title else file_title
                parent_title = ""
                for i in range(current_title_level - 1, 0, -1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break
                parent_title = parent_title if parent_title else file_title
                #对body_content中的表格进行文本化
                body_content = MarkdownTableLinearizer.process(body_content)
                #构造section
                section_list.append({
                    "file_title": file_title,
                    "parent_title": parent_title,
                    "title": title,
                    "body": body_content
                })

        content_lines = md_content.split("\n")
        #定义md文档标题的正则表达式
        title_pattern = re.compile(r"^\s*(#{1,6})\s+(.+)")
        #遍历md文档内容的每一行
        for line in content_lines:
            line = line.strip()
            if line.startswith("```") or line.startswith("~~~"):
                is_in_code_block = not is_in_code_block
            #匹配标题行
            match = title_pattern.match(line) if not is_in_code_block else None
            if match:
                # 将current_section中的内容构建成一个section
                _get_section()
                current_section = []
                current_title_level = len(match.group(1))
                current_title = line  # 标题 (带#)
                hierarchy[current_title_level] = current_title
                for i in range(current_title_level+1, len(hierarchy)):
                    hierarchy[i] = ""
            else:
                current_section.append(line)
        #构建最后一个section
        _get_section()

        return section_list

    def _log_summary(self, raw_content, section):
        line_content = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {line_content}")
        self.logger.info(f"最终切分章节数: {len(section)}")
        self.logger.info(f"最大切片长度: {self.config.max_content_length}")

        if section:
            self.logger.info("章节预览：")
            for i, sec in enumerate(section[:5]):
                title = sec.get("title", "")[:50]
                self.logger.info(f" {i + 1}. {title}...")
            if len(section) > 5:
                self.logger.info(f"...还有{len(section) - 5}个章节")

    def _split_and_merge_sections(self, section_list:List[Dict[str, Any]])->List[Dict[str, Any]]:
        self.log_step( "对按照标题切分后的块进行二次切分与合并处理")

        #对长的section进行二次切分
        splitted_section_list = self._split_long_sections(section_list)

        #对短的section进行合并
        final_section_list = self._merge_short_sections(splitted_section_list)

        #返回处理后的section列表
        return final_section_list

    def _split_long_sections(self, section_list:List[Dict[str, Any]])->List[Dict[str, Any]]:
        self.log_step("对长块进行二次切分")

        splitted_section_list = []
        for section in section_list:
            file_title = section.get("file_title", "")
            parent_title = section.get("parent_title", "")
            title = section.get("title", "")
            body_content = section.get("body", "")
            total_len = len(f"{title}\n\n{body_content}")
            if total_len > self.config.max_content_length:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.max_content_length - len(title + "\n\n"),
                    chunk_overlap=0,
                    separators=["\n\n", "\n", "。", "！", "？",".","!","?"," ",""],
                    keep_separator=False    #切分后是否保留分隔符
                )
                texts = text_splitter.split_text(body_content)
                #对切分后的texts列表中的每个文本都构造一个section
                for i, text in enumerate(texts):
                    splitted_section_list.append({
                        "file_title": file_title,
                        "parent_title": parent_title,
                        "title": title,
                        "body": text,
                        "part": i+1
                    })
            else:
                splitted_section_list.append(section)

        return splitted_section_list

    def _merge_short_sections(self, splitted_section_list:List[Dict[str, Any]])->List[Dict[str, Any]]:
        self.log_step("对短块进行合并处理")

        final_section_list = []
        if not splitted_section_list:
            return final_section_list
        #获取第一个section
        current_section = splitted_section_list[0]
        #从splitted_section_list中取出下一个section
        for i in range(1, len(splitted_section_list)):
            next_section = splitted_section_list[i]
            #判断是否同源
            is_same_parent = (current_section.get("parent_title") == next_section.get("parent_title") and current_section.get("title") == next_section.get("title"))
            #判断current_section的长度是否小于最小阈值
            current_section_len = len(current_section.get("title") + "\n\n" + current_section.get("body"))
            #是否合并
            if is_same_parent and current_section_len < self.config.min_content_length:
                #current_section["body"] = current_section.get("title") + "\n\n" + current_section.get("body") + "\n\n" + next_section.get("title") + "\n\n" + next_section.get("body")
                current_section["body"] = (current_section.get("body", "") + "\n\n" + next_section.get("title", "") + "\n\n" + next_section.get("body", ""))
                #current_section["title"] = current_section["parent_title"]
                if "part" in next_section:
                    current_section["part"] = next_section["part"]
            else:
                final_section_list.append(current_section)
                current_section = next_section
        final_section_list.append(current_section)
        #对final_section_list中带有part的section重新进行part编号
        order_dict = {}
        result_list = []
        for section in final_section_list:
            if "part" in section:
                title = section.get("parent_title")
                part_index = order_dict.get(title, 0) + 1
                section["part"] = part_index
                order_dict[title] = part_index
            result_list.append(section)

        return result_list

    def _assmble_chunks(self, final_section_list:List[Dict[str, Any]])->List[Dict[str, Any]]:
        self.log_step("对切分后的结果进行组装")

        chunks = []
        for section in final_section_list:
            file_title = section.get("file_title","")
            parent_title = section.get("parent_title","")
            title = section.get("title","")
            body = section.get("body","")
            content = f"{title}\n\n{body}"
            chunk = {
                "file_title": file_title,
                "parent_title": parent_title,
                "title": title,
                "content": content
            }
            if "part" in section:
                chunk["part"] = section.get("part")
            chunks.append(chunk)

        return chunks

    def _backup_chunks(self, state, sections):
        self.log_step("备份chunks")
        # 优先使用 file_dir，兼容 local_dir
        local_dir = state.get("file_dir", state.get("local_dir", ""))
        if not local_dir:
            self.logger.debug("未设置 file_dir/local_dir，跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=4)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


#测试
if __name__ == "__main__":
    setup_logging(logging.INFO)
    node = DocumentSpliterNode()
    state = {
    "task_id": "",
    "is_pdf_read_enabled": True,
    "is_md_read_enabled": False,
    "file_dir": "D:\\Python\\shopkeeper_brain\\knowledge\\processor\\import_process\\import_files",
    "import_file_path": "D:\\Python\\shopkeeper_brain\\knowledge\\processor\\import_process\\import_files\\万用表RS-12的使用.pdf",
    "pdf_path": "D:\\Python\\shopkeeper_brain\\knowledge\\processor\\import_process\\import_files\\万用表RS-12的使用.pdf",
    "md_path": "D:\\Python\\shopkeeper_brain\\knowledge\\processor\\import_process\\import_files\\万用表RS-12的使用\\auto\\万用表RS-12的使用.md",
    "file_title": "万用表RS-12的使用",
    "md_content": "# 使用说明书\n\n编号: 123-1939\n\n# 数字万用表\n\n![**内容总结：**\n这张图片展示了一个数字万用表的正面面板线稿图。主要特征包括：\n1.  **显示屏**：顶部有一个液晶显示屏，显示着“-1.8.8.8”（通常为开机自检或全显状态）。\n2.  **功能按键**：屏幕下方有“HOLD”保持键和背光键（太阳图标）。\n3.  **功能选择旋钮**：中央是一个大的旋转开关，周围标注了各种测量档位，包括直流/交流电压（V）、直流/交流电流（A）、电阻（Ω）、二极管/通断测试以及电池测试（BATT），还有“OFF”关机档。\n4.  **测量插孔**：底部有三个插孔，分别标记为“10A”、“COM”（公共端）和“VΩmA”。\n5.  **其他标识**：右下角有“CE”认证标志和“CAT III 600V”安全等级标识。\n\n**标题：**\n数字万用表面板示意图](http://192.168.10.102:9000/knowledge-base-v1/万用表RS-12的使用/15a273d07e22f1f6f04fba3abc323227581a67501b4b91c67c953524eb454184.jpg)\n\n# 安全手册\n\n为了您的安全，请在使用本仪表之前仔细阅读该手册:使用本表时，请勿将输入的测量值超出其所允许的量程范围。\n\n<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>输入量程</td></tr><tr><td rowspan=1 colspan=1>功能</td><td rowspan=1 colspan=1>最大输入</td></tr><tr><td rowspan=1 colspan=1>交/直流电压</td><td rowspan=1 colspan=1>直流/交流电压600V</td></tr><tr><td rowspan=1 colspan=1>直流/交流电压</td><td rowspan=1 colspan=1>直流/交流电压600V,200Vrms用于200mV量程</td></tr><tr><td rowspan=1 colspan=1>mA直流</td><td rowspan=1 colspan=1>200mA250V快速熔断保险丝</td></tr><tr><td rowspan=1 colspan=1>A DC</td><td rowspan=1 colspan=1>10A250V快速熔断保险丝(最多每15分钟，需时30秒)</td></tr><tr><td rowspan=1 colspan=1>电阻,短路测试</td><td rowspan=1 colspan=1>250Vrms,最多15秒</td></tr></table>\n\n2. 在测量高压电路时，请严格注意个人及设备的安全防护措施。\n\n3. 若负极端口（COM）电压超出500V以上接地电压，请勿进行电压测试。\n\n4. 若功能开关置于电流，电阻或二极管位置时，请勿将表笔与电路相连接，否则会损坏仪表。\n\n5. 进行电阻或二极管测试时，应把电容放电并断开电源。\n\n6. 打开后盖，更换保险丝或电池之前，请关闭电源并取下表笔。\n\n7. 请勿使用仪表，直到电池盖和保险丝盖装好，螺丝拧紧。\n\n# 安全标识\n\n![根据图片视觉内容（黑色三角形内含感叹号）以及文档标题“安全标识”和上下文（涉及电压危险、仪表损坏等安全提示），该图片是通用的安全警告符号。在安全标识标准（如ISO 7010或GB 2894）中，三角形感叹号通常代表“警告”或“注意”，用于提醒人们注意潜在危险或需要参照说明书操作。\n\n结合上下文中的“WARNING”以及安全标识的常规命名，该图片的精准中文标题为：\n\n**警告标志** （或 **警告** / **注意标志**）\n\n**最终生成标题：**\n警告标志](http://192.168.10.102:9000/knowledge-base-v1/万用表RS-12的使用/d31832baa1ae433f2afd951b33f131b9b3d3ad690eb0d1e997f8636432e25a95.jpg)\n\n表明此操作须参照说明书进行。\n\nWARNING 表明此处可能出现危险电压，请避开以免导致死亡或严重伤害。\n\n表明此处可能出现危险电压，请避开以免导致仪表的损坏。\n\n请勿连接到500VAC或VDC的电路上。\n\n![最大电压、危险电压及双重绝缘安全标识\n\n**解析：**\n根据提供的上下文和图片内容，这张图片实际上垂直排列了三个不同的安全标识，分别对应上下文中的不同描述：\n1.  **顶部标识（接地符号 + MAX）：** 对应上文“请勿连接到500VAC或VDC的电路上”。这表示**最大电压**限制（Maximum Voltage），即该端子或电路允许的最大电压值，超过此值（如500V）禁止连接。\n2.  **中间标识（闪电符号）：** 对应下文“表明此端口可能出现危险电压”以及上文的“WARNING 表明此处可能出现危险电压”。这表示**危险电压**警告（High Voltage / Shock Hazard），提示用户注意电击风险。\n3.  **底部标识（双重正方形框/回字形）：** 对应下文“双绝缘保护”。这表示**双重绝缘**（Double Insulation / Class II Equipment），说明设备具有双重绝缘保护，无需接地保护。\n\n因此，标题应概括这三个主要标识内容。\n\n**标题：** 最大电压、危险电压及双重绝缘安全标识](http://192.168.10.102:9000/knowledge-base-v1/万用表RS-12的使用/5c4d0b72ed07b882b48ed51d0bb940e92884f4ee557d6b99e99446bb97627122.jpg)\n\n表明此端口可能出现危险电压。\n\n双绝缘保护。\n\n# 控制与端口\n\n1.LCD液晶显示  \n2.功能选择转盘  \n3.10A端口  \n4.COM端口  \n5.正极端口  \n6.数据保持按键  \n7.背光按键\n\n# 规格\n\n![数字万用表前面板结构及功能说明图](http://192.168.10.102:9000/knowledge-base-v1/万用表RS-12的使用/6638e61dca3593c71c8a94ff1baa75fcb78802007317d62c4399cacff839f6b9.jpg)\n\n# 功能符号指示\n\n•))) 蜂鸣指示  \n二极管测试指示  \nµ micro (电流范围)  \nm milli ( 电压/电流范围)  \nk kilo (电阻范围)  \nVDC 直流电压  \nVAC 交流电流  \nADC 直流电流  \nBAT 电池电量不足指示\n\n<table><tr><td rowspan=1 colspan=1>功能</td><td rowspan=1 colspan=1>量程</td><td rowspan=1 colspan=1>分辨率</td><td rowspan=1 colspan=1>精确度</td></tr><tr><td rowspan=5 colspan=1>直流电压</td><td rowspan=1 colspan=1>200mV</td><td rowspan=1 colspan=1>0.1mV</td><td rowspan=3 colspan=1>±(0.5% reading +2 digits)</td></tr><tr><td rowspan=1 colspan=1>2000mV</td><td rowspan=1 colspan=1>1mV</td></tr><tr><td rowspan=1 colspan=1>20V</td><td rowspan=1 colspan=1>0.01V</td></tr><tr><td rowspan=1 colspan=1>200V</td><td rowspan=1 colspan=1>0.1V</td><td rowspan=2 colspan=1>± (0.8% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>600V</td><td rowspan=1 colspan=1>1V</td></tr><tr><td rowspan=2 colspan=1>交流电压</td><td rowspan=1 colspan=1>200V</td><td rowspan=1 colspan=1>0.1V</td><td rowspan=2 colspan=1>± (1.2% reading + 10 digits50/60Hz)</td></tr><tr><td rowspan=1 colspan=1>600V</td><td rowspan=1 colspan=1>1V</td></tr><tr><td rowspan=4 colspan=1>直流电流</td><td rowspan=1 colspan=1>2000μA</td><td rowspan=1 colspan=1>1uA</td><td rowspan=2 colspan=1>± (1.0% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>20mA</td><td rowspan=1 colspan=1>10uA</td></tr><tr><td rowspan=1 colspan=1>200mA</td><td rowspan=1 colspan=1>100uA</td><td rowspan=1 colspan=1>± (1.2% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>10A</td><td rowspan=1 colspan=1>10mA</td><td rowspan=1 colspan=1>± (2.0% reading + 2 digits)</td></tr><tr><td rowspan=5 colspan=1>电阻</td><td rowspan=1 colspan=1>200Ω</td><td rowspan=1 colspan=1>0.1Ω</td><td rowspan=4 colspan=1>±(0.8% reading +2 digits)</td></tr><tr><td rowspan=1 colspan=1>2000Ω</td><td rowspan=1 colspan=1>1Ω</td></tr><tr><td rowspan=1 colspan=1>20kΩ</td><td rowspan=1 colspan=1>0.01kΩ</td></tr><tr><td rowspan=1 colspan=1>200kΩ</td><td rowspan=1 colspan=1>0.1kΩ</td></tr><tr><td rowspan=1 colspan=1>2000kΩ</td><td rowspan=1 colspan=1>1kΩ</td><td rowspan=1 colspan=1>±(1.0% reading +2 digits)</td></tr><tr><td rowspan=2 colspan=1>电池</td><td rowspan=1 colspan=1>9V</td><td rowspan=1 colspan=1>10mV</td><td rowspan=2 colspan=1>±(1.0% reading +2 digits)</td></tr><tr><td rowspan=1 colspan=1>1.5V</td><td rowspan=1 colspan=1>1mV</td></tr></table>\n\n注意: 精确度规格由两种因素组成。● $\\%$ reading) –测量电路的精确度。● ( $^ +$ digits) –数位转换器条码的精确度。注意: 精确度在 $6 5 ^ { \\circ } \\mathsf { F }$ 至 $8 3 ^ { \\circ } \\mathsf { F }$ ( $\\underline { { 1 8 ^ { \\circ } } } \\mathsf { C }$ 至 $2 8 ^ { \\circ } \\mathsf { C } )$ ，湿度低于 $7 5 \\%$ RH时得出。\n\n# 技术指标说明\n\n二极管测试 测试电流最大值1mA, 开路电压 2.8V DC典型值  \n短路蜂鸣测试 若电阻小于30时产生蜂鸣  \n电池测试电流 9V (6mA)；1.5V (100mA)  \n输入阻抗 >1MΩ  \n交流电压频宽 45Hz～450Hz  \nDCA电压跌路测试 200mV  \n显示 3 ½ 数位，2000位液晶显示，1.1”数位  \n超量程提示 以“1”表示  \n极性 自动(正极无显示);负极显示(-)  \n测量率 正常情况下每秒2次  \n低电池提示 电池电压不足时，显示BAT符号  \n电池 一粒9V (NEDA 1604) 电池  \n保险丝 mA, µA 量程;0.2A/250V 快速熔断保险丝，A 档量程10A/250V快速熔断保险丝  \n操作环境 32°F～122°F (0°C～50°C)  \n储存温度 -4°F～140°F (-20°C～60°C)  \n相对湿度 <70% 操作, $< 8 0 \\%$ 储存  \n室内使用,最高海拔 7000英尺(2000米)  \n重量 255g  \n尺寸 150mm x 70mm x 48mm  \n安全认证 室内使用，符合过电压类别II  \n污染级别 2\n\n# 电池安装\n\n警告: 为防触电, 打开电池后盖前后，请勿操作仪表并把表笔与电源断开。\n\n1. 把表笔与仪表断开。  \n2. 用螺丝刀拧开电池后盖上的螺母。  \n3. 正确安装电池，正负极应一致。  \n4. 盖上电池后盖并拧紧螺丝钉。\n\n警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。\n\n注意: 若仪表出现工作不正常，请检测保险丝和电池是否完好以及是否放在正确的位置。\n\n# 操作指导\n\n# 数值保持按键HOLD\n\n保持键允许仪表固定测量值以供参考：\n\n1. 按下“HOLD”键保持读数， 同时出现“HOLD”字符  \n2. 再次按下“DATA HOLD”键 切换至正常操作\n\n# 背光灯键（BACKLIGHT）\n\n1. 按下背光灯键开启背光灯。  \n2. 再次按背光灯键关闭背光灯。\n\n警告：小心触电，高压电流十分危险，应小心操作。\n\n1. 为了节省电池损耗，使用后请将旋钮调至“OFF”档。\n\n2. 若测量过程中显示屏出现“OL”，表明测量值超出所选档位，应改选更高档。\n\n注意:在某些低交直流电压档位内，若表笔与被测物断开，显示屏将出现任意不稳定数值。该现象由高输入灵敏度所致。若接通电路，可读到稳定准确的数值。\n\n# 测量非接触交流电压\n\n警告: 为了防止电击，请在使用前，确保正确使用此非接触交流电压测电笔。\n\n1. 让其探头靠近或插入火线的输出插座孔时。\n\n2. 如果火线带有220V交流电输出，指示灯就会被点亮。\n\n注意: 如果是零线和火线缠绕在一起时，此时测试要将两线分开，来进行火线与零线的区分。注意: 此非接触交流电压测电笔设计为高度灵敏探测.当遇到静电或其它能带电体时，可能指示灯也会亮起或瞬间闪烁，这属于正常现象。\n\n直流电压测量\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n1. 将功能转盘置于V DC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物,确保极性正确(红色连正极,黑色连负极)。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值。若极性颠倒，数值前将显示负号。\n\n# 交流电压测量\n\n注意：在10A情况下测量时间不能超过30秒，否则将可能损坏仪表或表笔。\n\n![交流电压测量示意图\n\n**简要总结：**\n该图片展示了使用数字万用表测量交流电压（符号V）的接线方法。图中数字万用表的旋钮指向交流电压档，两根表笔（连接在COM和V/Ω插孔）并联在待测电压源（V）两端，屏幕显示读数（此处为0.00或初始状态），直观地演示了交流电压测量的标准操作和连接方式。上文提到的10A注意事项通常属于电流测量章节的安全提示，作为通用警告出现，而图片内容明确对应文档标题“交流电压测量”。\n\n**精准标题：**\n交流电压测量示意图（或：万用表测量交流电压接线图）](http://192.168.10.102:9000/knowledge-base-v1/万用表RS-12的使用/74d0155bc7ba7e29f7e906e09541410b68efcb115cea244d1e66699715954416.jpg)\n\n# 直流电流测量\n\n警告：谨防触电。\n\n若表笔长度不够不能接触到某些240V用具插座的带电部位，则可能出现插座有电而读到的数值却为0的情况。因此若无电压显示，应检查表笔是否接触到了插座内的金属接口。\n\n![根据图片视觉内容（数字万用表、表笔连接至标有“V”的电压符号）以及上下文信息（提及240V插座、电压显示、表笔接触检查、瞬间电压损坏仪表），这张图片主要展示了使用万用表进行电压测量的场景及相关的操作注意事项（尽管文档大标题为“直流电流测量”，但上下文和图片内容明显属于电压测量章节，可能是文档结构问题或通用警告）。\n\n图片展示了万用表通过表笔并联在电路两点间（标有V）测量电压。\n\n**标题：万用表测量电压示意图**\n\n(或者更具体一点结合上下文：**万用表电压测量及表笔连接示意图**)\n\n**最终生成标题：**\n万用表测量电压示意图](http://192.168.10.102:9000/knowledge-base-v1/万用表RS-12的使用/0cfb9ae7222f3704da1e70327a322f7e1b7d550827e58e683eacd2e6e4346043.jpg)\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n1. 将黑色表笔插入负极COM端口。  \n2. 测量直流200mA 以下的电流,将功能转盘置于最高DC mA档位，并将红色表笔插入mA端口。  \n3. 测量直流10A时,将功能转盘置于10A档位，并将红色表笔(10A)端口。  \n4. 断开被测电路的电源。在你想测量电流的位置打开电路绝缘层。  \n5. 将黑色表笔接触被测电路的负极，红色表笔接触被测电路正极。  \n6. 接通电源。  \n7. 在显示屏上读取读数。进行mA DC测量时,不断重调功能转盘至低mA DC档位获得高分辨率读数.读数由精确的小数点和数值表示。\n\n1. 将功能转盘置于V AC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值和(AC,V等)符号。\n\n在显示屏上读取电压数据。不断重调功能转盘至低交流电压档位获得高分辨率读数。读数由精确的小数点和数值表示。\n\n![基于提供的上下文信息（主要描述将万用表设置为V AC档位并读取电压值）以及图片视觉内容（万用表并联在负载两端，显示读数，电路图中有电压源符号V），尽管文档大标题为“直流电流测量”（可能存在文档分类错误或章节标题不匹配），图片和上下文内容明显是在描述**交流电压测量**的过程。\n\n以下是生成的标题：\n\n**交流电压测量**\n\n(或者更详细一点：**万用表交流电压测量**)\n\n**简要总结：**\n该图片展示了使用数字万用表测量交流电压的示意图。万用表功能盘置于V AC档位，表笔（红黑）并联连接在负载（Load）两端，显示屏上显示读数12.0，用于演示如何读取交流电压数据。\n\n**最终标题：**\n**交流电压测量**](http://192.168.10.102:9000/knowledge-base-v1/万用表RS-12的使用/b4cf6778e3c0504f6ad46c8be56fc27646f1e99fc4f5285c6ac74e960b46333f.jpg)\n\n# 电阻测量\n\n警告: 为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线。\n\n1. 将功能转盘置于最高电阻Ω位置.\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口\n\n3. 把表笔接触被测电路或元件。测试时最好断开电路的一端，以使剩余的电路不会干扰被测电阻数值。\n\n4. 读取显示屏上读数，然后将功能转盘调至最低电阻Ω档位，通常大于实际电阻或预测电阻.读数由精确的小数点和数值表示。\n\n![万用表测量电阻示意图](http://192.168.10.102:9000/knowledge-base-v1/万用表RS-12的使用/94d8ba0bd8763c180d4e9f54de465ca9042bd71319a4382281bbaeece25982cd.jpg)\n\n# 短路蜂鸣测试\n\n警告：请不要在接通电源的情况下进行在线短路蜂鸣测试以免触电。\n\n1. 将功能键转盘置于 位置。  \n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口。  \n3. 把表笔与被测物体相接触。  \n4. 当电阻小于30时Ω，仪表会发出蜂鸣.如果是开路，显示屏将显示“1”  \n字符。\n\n# 二极管测试\n\n1. 将黑色表笔插入负极COM 端口，红色表笔插入正极V端口。2. 将功能转盘置于 位置。3. 把表笔与二极管相接触，正向电压将显示400 至 700mV.反向电压显示“ 1”符号.短路时将显示接近 0V，开路时会在两种极性上显示“1”符号。\n\n# 电池测试\n\n1. 将黑色表笔插入负极COM端口，红色表笔插入正极V 端口。\n\n2. 使用功能选择键，选择1.5V 或 9V 电池档位。\n\n3. 将红色表笔接触电池正极，将黑色表笔接触电池负极。\n\n4. 在显示屏上读取数值。\n\n<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>良好</td><td rowspan=1 colspan=1>较弱</td><td rowspan=1 colspan=1>坏的</td></tr><tr><td rowspan=1 colspan=1>9V电池:</td><td rowspan=1 colspan=1>&gt;8.2V</td><td rowspan=1 colspan=1>7.2至8.2V</td><td rowspan=1 colspan=1>&lt;7.2V</td></tr><tr><td rowspan=1 colspan=1>1.5V电池：</td><td rowspan=1 colspan=1>&gt;1.35V</td><td rowspan=1 colspan=1>1.22至1.35V</td><td rowspan=1 colspan=1>&lt;1.22V</td></tr></table>\n\n# 更换电池\n\n警告：为防触电，打开电池后盖前后，请勿操作仪表并把表笔与电源断开。  \n1. 当电池电压不足时，显示屏上会出现“BAT”符号，此时应更换电池。  \n2. 按下面的步骤安装电池。  \n3. 妥善处理废电池。\n\n警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。\n\n# 更换保险丝\n\n警告:为防触电，在打开保险丝门之前，请把表笔和电源断开。\n\n1. 把表笔与仪表及其它被测物断开。  \n2. 用螺丝刀拧开保险丝门上的螺母。  \n3. 轻轻取出废旧的保险丝。  \n4. 装入新的保险丝。  \n5. 使用正确型号与数值的保险丝(0.2A/250V) 快速熔断保险丝用于200mA  \n的量程，10A/250V 快速熔断保险丝用于10A的量程。  \n6. 盖回后盖，拧紧螺钉。\n\n警告: 为防触电，在保险盖盖紧前请勿操作仪表。",
    "chunks": [],
    "item_name": ""
}
    state = node(state)
    json_str = json.dumps(state, indent=4, ensure_ascii=False)
    print(json_str)
