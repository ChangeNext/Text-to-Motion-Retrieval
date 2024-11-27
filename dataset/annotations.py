import json

def filter_annotations(annotations):
    filtered_annotations = {}
    for key, val in annotations.items():
        annots = val.pop("annotations")
        if annots:
            # 첫 번째 어노테이션만 선택
            first_annot = annots[0]
            filtered_annotations[key] = {
                "start": first_annot["start"],
                "end": first_annot["end"]
            }

    return filtered_annotations

def rearrange_annotations(annotations):
    rearranged_annotations = {}
    for key, val in annotations.items():
        annots = val["annotations"]
        if annots:
            start = annots[0]["start"]  
            end = annots[0]["end"]      
            text_list = [annot["text"] for annot in annots]
            rearranged_annotations[key] = {
                "start": start,
                "end": end,
                "text": text_list
            }
    return rearranged_annotations

with open("/data/motion/BLMP/dataset/humanml3d/annotations_1.json", "rb") as ff:
    original_data = json.loads(ff.read())
filtered_data = rearrange_annotations(original_data)

with open("/data/motion/BLMP/dataset/humanml3d/filtered_annotations.json", "w") as outfile:
    json.dump(filtered_data, outfile, indent=4)