from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info
#from .generate_qa import (
#    draw_detections,
#    extract_frame_info,
#    extract_kart_objects,
#    extract_track_info
#)

def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    #raise NotImplementedError("Not implemented")

    # Step 1: Extract all kart objects from the frame
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)

    # If no karts are detected, return empty captions
    if not karts:
        return []
    
    # Step 2: Identify the ego kart (the one closest to center)
    ego = next( k for k in karts if k.get("is_center_kart", False))

    # Step 3: Get track name from metadata
    track = extract_track_info(info_path)

    # List to store all generated captions  
    captions = []

    # Step 4: Basic captions about ego kart
    captions.append(f"{ego['kart_name']} is positioned as the ego kart on the {track} track.")
    captions.append(f"The ego kart on the {track} track is {ego['kart_name']}.")

    # Step 5: Count number of other karts
    num_others = len(karts) - 1

    # Add captions describing number of karts
    captions.append(f"There are {num_others} other karts present with {ego['kart_name']} on the {track} track.")
    captions.append(f"{ego['kart_name']} is navigating the {track} track alongside {num_others} other karts.")

    # Step 6: Get all karts except ego
    others = [k for k in karts if k["instance_id"] != ego["instance_id"]]

    # Step 7: Generate relative position captions
    for other in others:
        # Determine left/right position
        lr = "left" if other["center"][0] < ego["center"][0] else "right"

        # Determine front/back position
        fb = "ahead" if other["center"][1] < ego["center"][1] else "behind"

        # Add captions describing relative position
        captions.append(f"{other['kart_name']} is positioned {fb} and to the {lr} of the ego car")
        captions.append(f"To the {lr} and {fb} of {ego['kart_name']} is {other['kart_name']}")

    # Step 8: Combined summary caption (if multiple karts exist)
    if len(others) > 1:
        positions = []

        for other in others:
            lr = "left" if other["center"][0] < ego["center"][0] else "right"
            fb = "ahead" if other["center"][1] < ego["center"][1] else "behind"

            # Store formatted position string
            positions.append(f"{other['kart_name']} ({fb} - {lr})")

        # Add one combined descriptive caption
        captions.append(
            f"Kart positions relative to {ego['kart_name']}:"
            f"{', '.join(positions[:-1])} and {positions[-1]}"
        ) 
    
    # Step 9: Return all generated captions
    return captions

def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""

import json
from pathlib import Path
from tqdm import tqdm
import fire

def generate_all(
    input_dir: str = "data/train",
    output_file: str = "data/train/train_captions.json",
    max_views: int = 10,
):
    root = Path(input_dir).resolve()
    records = []

    for info_path in tqdm(root.rglob("*_info.json"), desc="Processing JSON files"):
        stem = info_path.stem.removesuffix("_info")

        for view in range(max_views):
            captions = generate_caption(str(info_path), view)
            if not captions:
                continue
            
            jpg_path = info_path.parent / f"{stem}_{view:02d}_im.jpg"
            rel_path = Path("train") / jpg_path.relative_to(root)

            records.extend(
                {"image_file": str(rel_path), "caption": caption}
                for caption in captions
            )
    
    Path(output_file).write_text(json.dumps(records, indent=2))
    print(f"Saved {len(records):,} captions -> {output_file}")

def main():
    fire.Fire({
        "check": check_caption
        #"generate_all": generate_all
    })


if __name__ == "__main__":
    main()
