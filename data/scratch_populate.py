import json
import os

new_data = [
    {
        'messages': [
            {'role': 'system', 'content': 'You are a BIM compliance reasoning assistant.'},
            {'role': 'user', 'content': 'Context: [{"text": "Beam YC-ST-SF-BIP is made of Wood. It requires a fire rating of 120 mins.", "meta": {"layer": "L1"}}]\n\nCompare the material of beams and columns and identify any non-compliant materials.'},
            {'role': 'assistant', 'content': 'Based on the context, the beam (YC-ST-SF-BIP) is made of Wood. Wood is generally NON_COMPLIANT if it fails to meet the 120 mins fire rating unless specially treated. (Note: Column data was not provided in the context to compare).'}
        ]
    },
    {
        'messages': [
            {'role': 'system', 'content': 'You are a BIM compliance reasoning assistant.'},
            {'role': 'user', 'content': 'Context: [{"text": "Wall W-12 has length 5500mm and a fire rating of 30 mins.", "meta": {"layer": "L1"}}, {"text": "Regulation 4.1: Walls exceeding 5000mm in length must have a fire rating of at least 60 mins.", "meta": {"layer": "L4"}}]\n\nWhich walls have a length greater than 5000mm and are they compliant with fire safety?'},
            {'role': 'assistant', 'content': 'Wall W-12 has a length of 5500mm, which is greater than 5000mm. According to Regulation 4.1, such walls require a fire rating of at least 60 mins. Since W-12 only has a fire rating of 30 mins, it is NON_COMPLIANT with fire safety.'}
        ]
    },
    {
        'messages': [
            {'role': 'system', 'content': 'You are a BIM compliance reasoning assistant.'},
            {'role': 'user', 'content': 'Context: [{"text": "Door D-01 is an evacuation door. It swings inward. Corridor width is 1100mm.", "meta": {"layer": "L1"}}, {"text": "Evacuation doors must swing outward. Corridors must be >1200mm wide.", "meta": {"layer": "L4"}}]\n\nAre the evacuation routes compliant? Explain the non-compliant elements.'},
            {'role': 'assistant', 'content': 'The evacuation routes are NON_COMPLIANT. \n1. Door D-01 swings inward, but regulations require evacuation doors to swing outward.\n2. The corridor width is 1100mm, which is less than the required minimum of 1200mm.'}
        ]
    }
]

file_path = 'data/finetuning_dataset.jsonl'
with open(file_path, 'a', encoding='utf-8') as f:
    for item in new_data:
        f.write(json.dumps(item) + '\n')
