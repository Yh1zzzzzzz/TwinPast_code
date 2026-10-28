# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Official realtime session; full-context cross-attention backend repair."""
from streaming_adapter import Adapter as Original
import attention_gqa as attention
class Adapter(Original):
    def __init__(self):
        super().__init__();attention.install()
    def answer_stream(self,job,prompt,emit):
        attention.reset_stats()
        result=super().answer_stream(job,prompt,emit)
        result.update(attention_backend='native Flash GQA for neutral mask; blocked exact online softmax otherwise',
                      attention_statistics=dict(attention.STATS),context_truncated=False)
        emit(dict(type='attention_backend_summary',**attention.STATS))
        return result
