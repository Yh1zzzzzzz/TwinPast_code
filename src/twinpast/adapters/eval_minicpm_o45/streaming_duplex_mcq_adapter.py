# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Official duplex history with explicit closed-set MCQ readout."""
import re
import torch
from streaming_duplex_qa_adapter import Adapter as TextQA

class Adapter(TextQA):
    def answer_from_cache(self,prompt,limit=128):
        e=self.duplex;tok=e.tokenizer;before=e.decoder.get_cache_length()
        letters=re.findall(r'^([A-Z])\. ',prompt,re.M)
        if len(letters)<2 or letters!=[chr(65+i) for i in range(len(letters))]:
            raise ValueError('Explicit contiguous option letters required for MCQ readout')
        prefix='The correct option is'
        candidates=[tok.encode(' '+letter,add_special_tokens=False) for letter in letters]
        if any(len(x)!=1 for x in candidates):raise ValueError('Option letters must be single tokens')
        for letter,tokens in zip(letters,candidates):
            if tok.encode(prefix,add_special_tokens=False)+tokens!=tok.encode(prefix+' '+letter,add_special_tokens=False):
                raise ValueError('MCQ prefix/candidate token boundary must be canonical')
        text=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,
            add_generation_prompt=True,enable_thinking=False)+prefix
        ids=tok(text,return_tensors='pt',add_special_tokens=False)['input_ids'].to('cuda:0');size=ids.shape[1]
        output=self.model.llm(input_ids=ids,past_key_values=e.decoder.cache,
            cache_position=torch.arange(before,before+size,device=ids.device),
            attention_mask=torch.ones((1,before+size),device=ids.device,dtype=torch.long),
            use_cache=True,return_dict=True)
        e.decoder.cache=output.past_key_values
        candidate_ids=[x[0] for x in candidates];logits=output.logits[0,-1].float()
        scores=logits[candidate_ids];chosen=int(scores.argmax());response=letters[chosen]
        choice=dict(candidate_letters=letters,candidate_token_ids=candidate_ids,candidate_logits=scores.cpu().tolist(),
            relative_candidate_probabilities=scores.softmax(-1).cpu().tolist(),selected_token_id=candidate_ids[chosen],
            unrestricted_next_token=tok.decode([int(logits.argmax())]),
            readout='constant assistant prefix + argmax over public legal option tokens; no GT/free-text remapping')
        return dict(response=response,raw_response='The correct option is '+response,
            raw_generated_token_ids=[candidate_ids[chosen]],output_tokens=1,query_template=text,
            kv_before_query=before,final_kv_tokens=e.decoder.get_cache_length(),mcq_readout=choice,
            selected_answer_committed_to_cache=False)

    def answer_stream(self,job,prompt,emit):
        result=super().answer_stream(job,prompt,emit)
        result.update(native_api='official MiniCPMODuplex observation + explicit forced-choice final text readout',
            query_sampling='one legal letter by model logits; no free text generation',
            query_readout_protocol='closed-set MCQ extension; not official duplex voice QA/free generation')
        return result
