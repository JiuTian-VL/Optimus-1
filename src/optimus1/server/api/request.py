from enum import Enum
from typing import Any, List

from pydantic import BaseModel


class MCRequest(BaseModel):
    rgb_images: List[Any]  # base64编码的图像信息

    done_imgs: List[Any] | None = None
    cont_imgs: List[Any] | None = None
    replan_imgs: List[Any] | None = None

    task_or_instruction: str  # 指令信息

    current_step: int = 0  # 当前步骤数

    history: List[str] | None = None  # 历史信息

    temperature: float = 0.2
    system_prompt: str | None = None

    type: str | None = None  # plan|action|replan|reflection
    error_info: str | None = None
    example: str | None = None
    graph: str | None = None
    visual_info: str | None = None


class MCResponse(BaseModel):
    response: Any = None

    status_code: int = 200
    message: str = ""

    appendix: Any | None = None