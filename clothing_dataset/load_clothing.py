from datasets import Dataset, DatasetDict, Value, Features, ClassLabel, Image, Sequence
import os
import json


def load_data_from_directory(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                file_data = {}
                file_data[
                    'image'] = f"/Users/adam/Downloads/clothing/dataset/{filename}"[
                               :-4] + "jpg"
                file_data["objects"] = json.load(file)
                data.append(file_data)
    return data


if __name__ == "__main__":
    # Path to the directory containing your output JSON files
    output_directory = '~/Downloads/clothing/adjusted-jsons'

    # Load the data
    data = load_data_from_directory(output_directory)
    categories = [
        "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket",
        "vest", "pants", "shorts", "skirt", "coat", "dress", "jumpsuit", "cape",
        "glasses", "hat", "headband, head covering, hair accessory", "tie", "glove",
        "watch", "belt", "leg warmer", "tights, stockings", "sock", "shoe",
        "bag, wallet", "scarf", "umbrella", "hood", "collar", "lapel", "epaulette",
        "sleeve", "pocket", "neckline", "buckle", "zipper", "applique", "bead", "bow",
        "flower", "fringe", "ribbon", "rivet", "ruffle", "sequin", "tassel", "earring",
        "ring", "bracelet", "necklace", "swimsuit", "underwear", "backpack"
    ]
    categories = ClassLabel(num_classes=len(categories), names=categories)

    features = Features({
        'image': Value('string'),
        "objects": Sequence(
            {'bbox_id': Value('int32'),
             'category': categories,
             'bbox': Sequence(Value('int32'), length=4),
             'area': Value('int32'),
             'genre': Value('string'),
             }

        )

    })

    dataset = Dataset.from_list(data, features=features)

    # Classify the full dataset as train split
    dataset_dict = DatasetDict({
        'train': dataset
    })

    print(dataset_dict["train"][:3])

    # dataset_dict = dataset_dict.cast_column("category", [categories])
    dataset_dict = dataset_dict.cast_column("image", Image())
    print(dataset_dict["train"][:3])


    # print(dataset_dict["train"][0]["image"].width)
    def add_width(batch):
        width = [img.width for img in batch["image"]]
        height = [img.height for img in batch["image"]]
        batch["width"] = width
        batch["height"] = height
        return batch


    dataset = dataset_dict["train"].map(add_width, batched=True)
    dataset_dict = DatasetDict({
        'train': dataset
    })
    print(dataset_dict["train"][:3])

    dataset_dict.push_to_hub("adam-narozniak/clothing", token="todo-give-your-token")
