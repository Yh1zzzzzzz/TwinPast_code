# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Explicit forced-choice final readout; no answer label enters candidate selection."""
import re
PREFIX='The correct option is'

def letters_from_prompt(prompt):
    letters=re.findall(r'^([A-Z])\. ',prompt,re.MULTILINE)
    if letters!=[chr(65+i) for i in range(len(letters))] or not 2<=len(letters)<=26:
        raise ValueError('MCQ readout requires ordered option letters in question prompt')
    return letters

def select(logits,tokenizer,prompt):
    import torch
    letters=letters_from_prompt(prompt);ids=[]
    for letter in letters:
        encoded=tokenizer.encode(' '+letter,add_special_tokens=False)
        if tokenizer.encode(PREFIX,add_special_tokens=False)+encoded!=tokenizer.encode(PREFIX+' '+letter,add_special_tokens=False):
            raise RuntimeError('MCQ candidate boundary is not canonical')
        if len(encoded)!=1:raise RuntimeError('Canonical option letter is not one token: '+letter)
        ids.append(encoded[0])
    values=logits[0,-1,ids].float();index=int(values.argmax())
    return dict(response=letters[index],selected_token_id=ids[index],candidate_letters=letters,
                candidate_token_ids=ids,candidate_logits=values.cpu().tolist(),
                relative_candidate_probabilities=values.softmax(-1).cpu().tolist(),
                unrestricted_next_token=tokenizer.decode([int(logits[0,-1].argmax())]),
                readout='constant assistant prefix + argmax over allowed letter tokens; no free-text remapping')
