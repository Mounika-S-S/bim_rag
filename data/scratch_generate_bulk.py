import json

more_data = [
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Column C-01 has a diameter of 400mm and compressive strength of 30MPa.\", \"meta\": {\"layer\": \"L1\"}}, {\"text\": \"Structural columns must have a minimum diameter of 450mm and 25MPa strength.\", \"meta\": {\"layer\": \"L4\"}}]\n\nIs Column C-01 structurally compliant?"},
            {"role": "assistant", "content": "Column C-01 is NON_COMPLIANT. While its compressive strength of 30MPa exceeds the required minimum of 25MPa, its diameter of 400mm is less than the regulation minimum of 450mm."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Slab S-Ground thickness is 150mm. Material is Reinforced Concrete.\", \"meta\": {\"layer\": \"L1\"}}]\n\nWhat is the thickness and material of the ground slab?"},
            {"role": "assistant", "content": "The ground slab (S-Ground) is made of Reinforced Concrete and has a thickness of 150mm."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Window W-05 is on the West facade. It is double-glazed. Thermal transmittance (U-value) is 1.8 W/m2K.\", \"meta\": {\"layer\": \"L1\"}}, {\"text\": \"West facade windows must have a maximum U-value of 1.5 W/m2K for energy efficiency.\", \"meta\": {\"layer\": \"L4\"}}]\n\nDoes Window W-05 meet energy efficiency standards?"},
            {"role": "assistant", "content": "Window W-05 is NON_COMPLIANT with energy efficiency standards. Its U-value is 1.8 W/m2K, which exceeds the maximum allowed U-value of 1.5 W/m2K for West facade windows."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Staircase ST-01 step height is 190mm and tread depth is 230mm.\", \"meta\": {\"layer\": \"L1\"}}, {\"text\": \"Public staircases must have a maximum step height of 180mm and minimum tread depth of 250mm.\", \"meta\": {\"layer\": \"L4\"}}]\n\nAre the staircase dimensions compliant for public use?"},
            {"role": "assistant", "content": "Staircase ST-01 is NON_COMPLIANT for public use. The step height (190mm) exceeds the maximum allowed (180mm), and the tread depth (230mm) is less than the required minimum (250mm)."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Room R-101 has a ceiling height of 2400mm and floor area of 12 sqm.\", \"meta\": {\"layer\": \"L1\"}}, {\"text\": \"Habitable rooms must have a minimum ceiling height of 2400mm.\", \"meta\": {\"layer\": \"L4\"}}]\n\nCheck if Room R-101 is compliant for habitation."},
            {"role": "assistant", "content": "Room R-101 is COMPLIANT for habitation. Its ceiling height is exactly 2400mm, which meets the minimum regulatory requirement."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Roof RF-Top has a pitch angle of 12 degrees. Material is Asphalt Shingles.\", \"meta\": {\"layer\": \"L1\"}}, {\"text\": \"Asphalt shingles require a minimum roof pitch of 15 degrees to prevent water ingress.\", \"meta\": {\"layer\": \"L4\"}}]\n\nIs the roof design appropriate for its material?"},
            {"role": "assistant", "content": "The roof design is NON_COMPLIANT. Roof RF-Top uses Asphalt Shingles but only has a pitch of 12 degrees. The regulations require a minimum pitch of 15 degrees for this material to prevent water ingress."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Fire Extinguisher FE-01 is located in Hallway H-1. Weight is 5kg.\", \"meta\": {\"layer\": \"L1\"}}]\n\nWhere is the fire extinguisher located?"},
            {"role": "assistant", "content": "Fire Extinguisher FE-01 is located in Hallway H-1."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Pipe P-Water-Main has a diameter of 50mm and is made of PVC.\", \"meta\": {\"layer\": \"L1\"}}]\n\nGive details regarding the main water pipe."},
            {"role": "assistant", "content": "The main water pipe (P-Water-Main) is made of PVC and has a diameter of 50mm."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Balcony B-03 has a railing height of 950mm.\", \"meta\": {\"layer\": \"L1\"}}, {\"text\": \"Balcony railings must be at least 1000mm high to prevent falls.\", \"meta\": {\"layer\": \"L4\"}}]\n\nIs the balcony railing safe according to regulations?"},
            {"role": "assistant", "content": "The balcony railing is NON_COMPLIANT and structurally unsafe according to regulations. Balcony B-03 has a railing height of 950mm, which fails to meet the minimum required height of 1000mm."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"HVAC Duct D-AC-01 has a flow rate of 500 CFM.\", \"meta\": {\"layer\": \"L1\"}}]\n\nWhat is the flow rate of the AC duct?"},
            {"role": "assistant", "content": "The HVAC Duct (D-AC-01) has a flow rate of 500 CFM."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Elevator L-01 has a weight capacity of 750kg and dimensions 1200x1400mm.\", \"meta\": {\"layer\": \"L1\"}}, {\"text\": \"Wheelchair accessible elevators must have minimum dimensions of 1100x1400mm and support at least 600kg.\", \"meta\": {\"layer\": \"L4\"}}]\n\nIs the elevator wheelchair accessible?"},
            {"role": "assistant", "content": "Yes, Elevator L-01 is COMPLIANT and wheelchair accessible. Its dimensions (1200x1400mm) satisfy the 1100x1400mm minimum requirement, and its 750kg capacity exceeds the 600kg minimum."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Beam B-55 has a clear span of 8000mm.\", \"meta\": {\"layer\": \"L1\"}}]\n\nWhat is the span of beam B-55?"},
            {"role": "assistant", "content": "Beam B-55 has a clear span of 8000mm."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Foundation Pad-01 has a depth of 900mm. Soil type is clay.\", \"meta\": {\"layer\": \"L1\"}}, {\"text\": \"Foundations in clay soil must have a minimum depth of 1000mm due to frost heave.\", \"meta\": {\"layer\": \"L4\"}}]\n\nDoes the foundation depth comply with soil constraints?"},
            {"role": "assistant", "content": "The foundation depth is NON_COMPLIANT. Pad-01 has a depth of 900mm in clay soil, failing to meet the minimum regulatory depth of 1000mm required to prevent damage from frost heave."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Wall W-party separates 2 distinct apartments. It has an acoustic rating of 45 dB.\", \"meta\": {\"layer\": \"L1\"}}, {\"text\": \"Party walls between apartments must have an acoustic rating of at least 50 dB.\", \"meta\": {\"layer\": \"L4\"}}]\n\nAre the acoustic properties of the party wall acceptable?"},
            {"role": "assistant", "content": "No, the party wall properties are NON_COMPLIANT. Wall W-party has an acoustic rating of 45 dB, which is lower than the mandatory minimum of 50 dB for walls separating distinct apartments."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Lighting fixture L-4 has an illuminance of 300 lux. Located in the Office area.\", \"meta\": {\"layer\": \"L1\"}}, {\"text\": \"General office areas must have a minimum illuminance of 500 lux.\", \"meta\": {\"layer\": \"L4\"}}]\n\nIs there enough light in the office?"},
            {"role": "assistant", "content": "No, the office lighting is NON_COMPLIANT. The lighting fixture (L-4) provides 300 lux, which falls short of the minimum 500 lux required for general office areas."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Sprinkler System SS-1 covers an area of 20 sqm per head.\", \"meta\": {\"layer\": \"L1\"}}, {\"text\": \"Sprinkler heads in residential hazards must cover a maximum of 12 sqm.\", \"meta\": {\"layer\": \"L4\"}}]\n\nAre the sprinklers appropriately spaced?"},
            {"role": "assistant", "content": "The sprinklers are NON_COMPLIANT. System SS-1 covers 20 sqm per head, heavily exceeding the maximum allowed coverage of 12 sqm for residential hazards."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Ramp R-Wheelchair has a slope of 1:10 and a length of 5 meters.\", \"meta\": {\"layer\": \"L1\"}}, {\"text\": \"Wheelchair access ramps must have a maximum slope of 1:12.\", \"meta\": {\"layer\": \"L4\"}}]\n\nIs the wheelchair ramp compliant?"},
            {"role": "assistant", "content": "Ramp R-Wheelchair is NON_COMPLIANT. Its slope is 1:10, which is steeper than the maximum permitted slope of 1:12 for wheelchair access ramps."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Parking space P-01 is 2.3m wide and 4.8m long.\", \"meta\": {\"layer\": \"L1\"}}, {\"text\": \"Standard parking spaces must be at least 2.4m wide and 4.8m long.\", \"meta\": {\"layer\": \"L4\"}}]\n\nDoes the parking space meet standard requirements?"},
            {"role": "assistant", "content": "Parking space P-01 is NON_COMPLIANT. While the length (4.8m) meets the requirements, the width (2.3m) is under the 2.4m minimum standard."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Emergency Generator EG-1 provides a backup power duration of 4 hours.\", \"meta\": {\"layer\": \"L1\"}}, {\"text\": \"High-rise emergency generators must maintain a backup duration of at least 8 hours.\", \"meta\": {\"layer\": \"L4\"}}]\n\nIs the emergency generator adequate?"},
            {"role": "assistant", "content": "The emergency generator is NON_COMPLIANT. EG-1 only provides 4 hours of backup power, failing the 8-hour minimum requirement for high-rise emergency scenarios."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Toilet T-Disabled has a grab bar mounted at 850mm height from the floor.\", \"meta\": {\"layer\": \"L1\"}}, {\"text\": \"Disabled toilets must have grab bars mounted between 800mm and 860mm from the floor.\", \"meta\": {\"layer\": \"L4\"}}]\n\nAre the grab bars in the disabled toilet correctly placed?"},
            {"role": "assistant", "content": "Yes, the grab bar placement is COMPLIANT. The height of 850mm falls well within the required regulatory range of 800mm to 860mm."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Smoke detector SD-04 is installed 8 meters away from the nearest wall in the main lobby.\", \"meta\": {\"layer\": \"L1\"}}, {\"text\": \"Smoke detectors must be installed no more than 7.5 meters from any wall.\", \"meta\": {\"layer\": \"L4\"}}]\n\nIs the smoke detector placement compliant?"},
            {"role": "assistant", "content": "The smoke detector placement is NON_COMPLIANT. SD-04 is 8 meters away from the wall, which exceeds the maximum allowable distance of 7.5 meters."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Facade Panel F-10 is made of Aluminum Composite Material (ACM) with a PE core.\", \"meta\": {\"layer\": \"L1\"}}, {\"text\": \"Building facades over 18m must use non-combustible material (PE cores are banned).\", \"meta\": {\"layer\": \"L4\"}}]\n\nAssuming the building is 25m tall, is the facade panel compliant?"},
            {"role": "assistant", "content": "The facade panel is NON_COMPLIANT. The building exceeds 18m, meaning PE core materials are strictly banned due to combustibility, yet Facade Panel F-10 utilizes a PE core ACM."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Fire door FD-Stairwell has a fire rating of 60 mins and no self-closing device.\", \"meta\": {\"layer\": \"L1\"}}, {\"text\": \"All fire doors must have a minimum 60 min fire rating and be fitted with an automatic self-closing device.\", \"meta\": {\"layer\": \"L4\"}}]\n\nIs the stairwell fire door compliant?"},
            {"role": "assistant", "content": "The fire door is NON_COMPLIANT. Even though it meets the 60-minute fire rating, it lacks the mandatory automatic self-closing device."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Retaining Wall RW-1 has weep holes spaced every 3.0 meters.\", \"meta\": {\"layer\": \"L1\"}}, {\"text\": \"Retaining walls must have weep holes spaced no more than 2.0 meters apart for proper drainage.\", \"meta\": {\"layer\": \"L4\"}}]\n\nDoes the retaining wall drainage comply?"},
            {"role": "assistant", "content": "The drainage is NON_COMPLIANT. Retaining Wall RW-1 has weep holes spread every 3.0 meters, exceeding the maximum allowed spacing of 2.0 meters for proper drainage."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a BIM compliance reasoning assistant."},
            {"role": "user", "content": "Context: [{\"text\": \"Solar Panel SP-Array occupies 40% of the flat roof area.\", \"meta\": {\"layer\": \"L1\"}}, {\"text\": \"Commercial buildings must allocate at least 30% of their flat roof area for renewable energy generation.\", \"meta\": {\"layer\": \"L4\"}}]\n\nIs the solar panel coverage compliant?"},
            {"role": "assistant", "content": "Yes, the coverage is COMPLIANT. The solar panels occupy 40% of the roof, safely exceeding the minimum requirement of 30% for commercial buildings."}
        ]
    }
]

file_path = 'data/finetuning_dataset.jsonl'
with open(file_path, 'a', encoding='utf-8') as f:
    for item in more_data:
        f.write(json.dumps(item) + '\n')

print(f"Appended {len(more_data)} records to {file_path}")
