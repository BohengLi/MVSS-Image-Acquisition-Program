from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import site
from typing import Any

import numpy as np


class CREStereoError(RuntimeError):
    pass


_CUDA_DLL_HANDLES: list[Any] = []


def _add_nvidia_cuda_dll_directories() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    candidates: list[Path] = []
    for base in [*site.getsitepackages(), site.getusersitepackages()]:
        nvidia_dir = Path(base) / "nvidia"
        if not nvidia_dir.exists():
            continue
        candidates.extend(path for path in nvidia_dir.glob("*/bin") if path.is_dir())
    for path in candidates:
        text = str(path)
        if text not in os.environ.get("PATH", ""):
            os.environ["PATH"] = text + os.pathsep + os.environ.get("PATH", "")
        try:
            _CUDA_DLL_HANDLES.append(os.add_dll_directory(text))
        except OSError:
            pass


@dataclass(frozen=True)
class CREStereoResult:
    disparity: np.ndarray
    model_path: str
    input_width: int
    input_height: int
    providers: list[str]
    input_names: list[str]
    output_names: list[str]


class CREStereoONNX:
    def __init__(self, model_path: str | Path, providers: list[str] | tuple[str, ...] | None = None) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise CREStereoError(f"CREStereo ONNX 模型文件不存在：{self.model_path}")
        _add_nvidia_cuda_dll_directories()
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise CREStereoError(
                "未安装 onnxruntime，无法使用 CREStereo。请执行 `python -m pip install -r requirements.txt`。"
            ) from exc

        requested = list(providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
        available = set(ort.get_available_providers())
        selected = [provider for provider in requested if provider in available]
        if "CPUExecutionProvider" not in selected and "CPUExecutionProvider" in available:
            selected.append("CPUExecutionProvider")
        if not selected:
            selected = list(available)
        if not selected:
            raise CREStereoError("onnxruntime 没有可用的 ExecutionProvider。")

        try:
            self.session = ort.InferenceSession(str(self.model_path), providers=selected)
        except Exception as exc:
            raise CREStereoError(f"CREStereo ONNX 模型加载失败：{exc}") from exc

        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        if len(self.inputs) not in {2, 4}:
            raise CREStereoError(f"CREStereo ONNX 输入数量应为 2 或 4，当前为 {len(self.inputs)}。")
        if not self.outputs:
            raise CREStereoError("CREStereo ONNX 模型没有输出节点。")

        self.input_names = [item.name for item in self.inputs]
        self.output_names = [item.name for item in self.outputs]
        shape = self.inputs[-1].shape
        try:
            self.input_height = int(shape[2])
            self.input_width = int(shape[3])
        except Exception as exc:
            raise CREStereoError(f"无法从 CREStereo ONNX 输入形状读取分辨率：{shape}") from exc
        if self.input_width <= 0 or self.input_height <= 0:
            raise CREStereoError(f"CREStereo ONNX 输入分辨率无效：{shape}")
        self.has_flow = len(self.inputs) == 4
        self.providers = list(self.session.get_providers())

    def predict(self, cv2: Any, left_bgr: np.ndarray, right_bgr: np.ndarray) -> CREStereoResult:
        if left_bgr.shape[:2] != right_bgr.shape[:2]:
            raise CREStereoError(f"CREStereo 左右图尺寸不同：{left_bgr.shape[:2]} vs {right_bgr.shape[:2]}")
        source_height, source_width = left_bgr.shape[:2]
        left_tensor = self._prepare_input(cv2, left_bgr, self.input_width, self.input_height)
        right_tensor = self._prepare_input(cv2, right_bgr, self.input_width, self.input_height)
        feed = {
            self.input_names[-2]: left_tensor,
            self.input_names[-1]: right_tensor,
        }
        if self.has_flow:
            feed[self.input_names[0]] = self._prepare_input(cv2, left_bgr, self.input_width // 2, self.input_height // 2)
            feed[self.input_names[1]] = self._prepare_input(cv2, right_bgr, self.input_width // 2, self.input_height // 2)

        try:
            output = self.session.run(self.output_names, feed)[0]
        except Exception as exc:
            raise CREStereoError(f"CREStereo ONNX 推理失败：{exc}") from exc

        disparity = np.squeeze(output).astype(np.float32)
        if disparity.ndim == 3:
            disparity = disparity[0]
        if disparity.ndim != 2:
            raise CREStereoError(f"CREStereo 输出维度应为 2D 视差图，当前为 {output.shape}。")
        if disparity.shape[:2] != (source_height, source_width):
            disparity = cv2.resize(disparity, (source_width, source_height), interpolation=cv2.INTER_LINEAR)
            disparity *= float(source_width) / max(float(self.input_width), 1.0)
        return CREStereoResult(
            disparity=disparity,
            model_path=str(self.model_path),
            input_width=int(self.input_width),
            input_height=int(self.input_height),
            providers=self.providers,
            input_names=self.input_names,
            output_names=self.output_names,
        )

    @staticmethod
    def _prepare_input(cv2: Any, bgr: np.ndarray, width: int, height: int) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (max(1, int(width)), max(1, int(height))), interpolation=cv2.INTER_AREA)
        tensor = resized.transpose(2, 0, 1)[np.newaxis, :, :, :]
        return tensor.astype(np.float32)
