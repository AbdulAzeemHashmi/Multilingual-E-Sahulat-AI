"""
Dataset Loader and Knowledge Base Builder
Extracts official canonical passages, metadata, and train_dev/test splits
from Multilingual E-Sahulat AI_Hackathon_Dataset.txt
"""

import os
import json
from typing import Dict, List, Any, Tuple


# Roman Urdu translations for canonical expected answers
ROMAN_URDU_ANSWERS = {
    "F01": "Pakistani aam passport tamam Pakistani shehriyon ko zaroori qawaid aur sharaait mukammal karnay ke baad jari kiya jata hai.",
    "F02": "18 saal ya us se baray applicants ko asoolan apna original valid CNIC/NICOP aur uski photocopy lani chahiye. Agar pehle passport bana hua ho to purana passport aur photocopy bhi sath layen.",
    "F03": "Sarkari, semi-government ya autonomous idaron ke mulazmeen ko apne mutalliqah department se No Objection Certificate (NOC) lana zaroori hota hai.",
    "F04": "Dual nationality walay shehri apna ghair mulki passport aur photocopy sath layen aur passport declaration form mein iska indraj zaroor karein.",
    "F05": "Passport gum honay ki soorat mein police report layen jis par puranay passport ka number wazeh darj ho.",
    "F06": "10 saal se kam umar bachon ke liye asal Child Registration Certificate (CRC / Form-B) ya Family Registration Certificate (FRC) aur aik photocopy zaroori hai.",
    "F07": "10 saal ya us se baray minor bachay ke liye fingerprints aur tasweer wala updated CRC ya Juvenile Card aur photocopy darkar hoti hai.",
    "F08": "Minor ke liye dono walidain ke original valid CNIC ya passport aur unki photocopies darkar hain. Bachay ke sath walidain mein se aik ya court ka muqarrar karda qanooni guardian hona zaroori hai.",
    "F09": "Walidain ki talaq ya alaidgi ki soorat mein adalat ke jari karda custody papers layen. Doosray walid/walida ki ijazat ya stamp paper par tasdeeq shuda ijazat nama darkar ho sakta hai.",
    "F10": "Agar walidain mein se kisi aik ki wafat ho chuki ho to unka original death certificate aur photocopy pesh karna hogi.",
    "F11": "Aam machine readable passport (MRP) 5 saal ya 10 saal ki validity ke liye jari kiya jata hai.",
    "F12": "15 saal se kam umar bachon ko qawaid ke tehat sirf 5 saal ki validity wala passport jari kiya jata hai.",
    "F13": "Normal MRP processing ke liye kamyab processing ke baad 21 working days lagtay hain; courier ka waqt is ke ilawa hota hai.",
    "F14": "Urgent MRP processing ke liye kamyab processing ke baad 5 working days lagtay hain.",
    "F15": "Fast Track passport processing ke liye 2 working days muqarrar hain.",
    "F16": "Passport expiry se pehle kisi bhi waqt renew karwaya ja sakta hai jab zaroorat ho.",
    "F17": "Passport renewal ke liye online apply kiya ja sakta hai agar purana passport expire ho chuka ho ya 1 saal se kam validity bachi ho.",
    "F18": "Online portal par renewal ki sahulat un afrad ke liye hai jinke passport mein koi tabdeeli (modification) na ho.",
    "F19": "Online portal par passport fees credit card, debit card ya digital banking se ada ki ja sakti hai.",
    "F20": "Online jama karwayi gayi fee kisi soorat wapis (refund) nahi hoti.",
    "F21": "Passport office mein pehla marhala photo session ka hota hai jahan digital picture li jati hai.",
    "F22": "Photo ke baad agli stage biometrics (fingerprints) ki hoti hai jahan ungliyon ke nishan liye jatay hain.",
    "F23": "Aakhri stage par Assistant Director ka interview hota hai jahan data verification hoti hai.",
    "F24": "Applicant ko apne CNIC/NICOP par darj mustaqil ya mojooda address ke mutabiq mutalliqah regional passport office jana hota hai.",
    "F25": "Pakistan mein aam taur par teen aqsam ke passport hain: Ordinary (Green), Official (Blue), aur Diplomatic (Red).",
    "F26": "Agar CNIC par sirf aik naam darj ho to e-Passport ke liye wahi naam 'Given Name' ke taur par likha jayega.",
    "F27": "E-Passport ke liye Fast Track sahulat dastiyab hai jo ke designated passport dafatir mein fraham ki jati hai.",
    "F28": "Agar passport ke safhay (pages) khatam ho jayen to naya passport banwana parta hai (fresh booklet issue hoti hai).",
    "F29": "Agar passport par naam tabdeel karwana ho to pehle NADRA CNIC par naam update karwana zaroori hai.",
    "F30": "Bachon ke liye alag passport zaroori hai; ab bachay ka naam walidain ke passport par darj nahi kiya ja sakta."
}


def load_raw_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Reads raw JSONL dataset from file."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith('{') and line_str.endswith('}'):
                records.append(json.loads(line_str))
    return records


def build_knowledge_base_and_splits(raw_filepath: str, output_dir: str = 'data') -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extracts canonical passage knowledge base and partitions into train_dev and test splits.
    """
    os.makedirs(output_dir, exist_ok=True)
    records = load_raw_dataset(raw_filepath)

    knowledge_base = {}
    train_dev_set = []
    test_set = []

    for r in records:
        iid = r.get('intent_id')
        if iid and iid not in knowledge_base:
            knowledge_base[iid] = {
                "passage_id": iid,
                "intent_id": iid,
                "topic": r.get('topic'),
                "gold_evidence_en": r.get('gold_evidence_en'),
                "gold_evidence_ur": r.get('gold_evidence_ur'),
                "expected_answer_en": r.get('expected_answer_en'),
                "expected_answer_ur": r.get('expected_answer_ur'),
                "expected_answer_roman_ur": ROMAN_URDU_ANSWERS.get(iid, r.get('expected_answer_en')),
                "source_url": r.get('source_url'),
                "domain": "Pakistan DGI&P Public Services"
            }

        split = r.get('split')
        if split == 'test':
            test_set.append(r)
        else:
            train_dev_set.append(r)

    kb_list = list(knowledge_base.values())
    kb_list.sort(key=lambda x: x['intent_id'])

    # Save to files
    kb_path = os.path.join(output_dir, 'knowledge_base.json')
    train_path = os.path.join(output_dir, 'train_dev.json')
    test_path = os.path.join(output_dir, 'test.json')

    with open(kb_path, 'w', encoding='utf-8') as f:
        json.dump(kb_list, f, indent=2, ensure_ascii=False)

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_dev_set, f, indent=2, ensure_ascii=False)

    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)

    print(f"Knowledge Base created: {len(kb_list)} canonical passages -> {kb_path}")
    print(f"Train/Dev Split: {len(train_dev_set)} records -> {train_path}")
    print(f"Test Split: {len(test_set)} records -> {test_path}")

    return kb_list, train_dev_set, test_set


if __name__ == '__main__':
    raw_path = 'Multilingual E-Sahulat AI_Hackathon_Dataset.txt'
    build_knowledge_base_and_splits(raw_path)
