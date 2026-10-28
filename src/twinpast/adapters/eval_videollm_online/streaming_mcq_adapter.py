# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Unchanged official LiveInfer history; final forced-choice QA."""
import torch
from streaming_adapter import Adapter as Original
from mcq_readout import PREFIX,select

class Adapter(Original):
    def __init__(self):
        super().__init__();live=self.live;original=live._call_for_response
        self.choice_record=None
        def respond(video_time,query):
            if query is None:return original(video_time,query)
            ids=self.processor.apply_chat_template([dict(role='user',content=query)],add_stream_query_prompt=True,
                add_generation_prompt=True,return_tensors='pt').to('cuda:0')
            prefix=torch.tensor([self.processor.encode(PREFIX,add_special_tokens=False)],device='cuda:0')
            ids=torch.cat([ids,prefix],1)
            output=self.model(input_ids=ids,past_key_values=live.past_key_values,use_cache=True,return_dict=True)
            live.past_key_values=output.past_key_values
            choice=select(output.logits,self.processor,query);choice['query_input_tokens']=ids.shape[1]
            # Commit the selected answer to the same persistent cache. As in the
            # released generator the final EOS itself remains in last_ids.
            selected=torch.tensor([[choice['selected_token_id']]],device='cuda:0')
            committed=self.model(input_ids=selected,past_key_values=live.past_key_values,use_cache=True,return_dict=True)
            live.past_key_values=committed.past_key_values
            live.last_ids=torch.tensor([[live.eos_token_id]],device='cuda:0')
            self.choice_record=choice
            return f'(Video Time = {video_time}s) User: {query}',f"(Video Time = {video_time}s) Assistant:{choice['response']}"
        live._call_for_response=respond

    def answer_stream(self,job,prompt,emit):
        self.choice_record=None
        result=super().answer_stream(job,prompt,emit)
        if self.choice_record is None:raise RuntimeError('No final forced-choice readout occurred')
        emit(dict(type='mcq_readout',**self.choice_record))
        result.update(mcq_readout=self.choice_record,query_decoder='forced choice; native streaming history unchanged',
                      native_api='official LiveInfer streaming + explicit forced-choice final text readout')
        return result

    def answer_offline(self,frames,meta,prompt,video):
        from preprocessing import resize_and_pad
        ids=self.processor.apply_chat_template([
            dict(role='system',content=self.live.system_prompt),dict(role='stream',num_frames=len(frames)),
            dict(role='user',content=prompt)],add_generation_prompt=True,return_tensors='pt').to('cuda:0')
        prefix=torch.tensor([self.processor.encode(PREFIX,add_special_tokens=False)],device='cuda:0')
        ids=torch.cat([ids,prefix],1);rgb=resize_and_pad(frames,resolution=384).to('cuda:0')
        with torch.inference_mode():
            output=self.model(input_ids=ids,frames=rgb,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True)
            choice=select(output.logits,self.processor,prompt)
        return dict(response=choice['response'],raw_response=PREFIX+' '+choice['response'],output_tokens=1,
            input_tokens=ids.shape[1],input_shapes=dict(frames=list(rgb.shape)),frame_visual_tokens=10,
            input_mode='uniform64 causal prefix + explicit forced-choice readout',mcq_readout=choice)
