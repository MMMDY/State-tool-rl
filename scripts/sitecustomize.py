"""Optional SGLang compatibility hooks for GPUs without matching sgl-kernel wheels."""

from __future__ import annotations

import os


if os.environ.get("SGLANG_FORCE_NATIVE_CUDA_OPS") == "1":
    from sglang.srt.custom_op import CustomOp

    _dispatch_forward = CustomOp.dispatch_forward

    def _dispatch_native_when_available(self):
        native = type(self).forward_native
        if native is not CustomOp.forward_native:
            return native.__get__(self, type(self))
        return _dispatch_forward(self)

    CustomOp.dispatch_forward = _dispatch_native_when_available

