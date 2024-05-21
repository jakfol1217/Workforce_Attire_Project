import json
import os

# Category list
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

# Spanish to English mappings
spanish_to_english = {
    "camisas": "shirt, blouse",
    "camisas ocultas debajo de la chaqueta": "shirt, blouse",
    "chaquetas": "jacket",
    "cazadoras": "jacket",
    "pantalones": "pants",
    "pantalones cortos": "shorts",
    "falda": "skirt",
    "abrigos": "coat",
    "vestidos": "dress",
    "monos": "jumpsuit",
    "gafas de sol": "glasses",
    "sombreros": "hat",
    "corbatas": "tie",
    "guantes y manopla": "glove",
    "relojes": "watch",
    "cinturones": "belt",
    "calcetines y medias": "tights, stockings",
    "zapatos": "shoe",
    "bolso": "bag, wallet",
    "carteras monederos": "bag, wallet",
    "bufandas": "scarf",
    "pendientes": "earring",
    "anillos": "ring",
    "pulseras": "bracelet",
    "collares": "necklace",
    "trajes de baño": "swimsuit",
    "ropa interior": "underwear",
    "mochilas": "backpack",
    "botas": "shoe"
}

# spanish_to_english = {
#     "zapatos": "shoe",
#     "camisas": "shirt, blouse",
#     "camisas ocultas debajo de la chaqueta": "shirt, blouse",
#     "pantalones": "pants",
#     "pantalones cortos": "shorts",
#     "falda": "skirt",
#     "abrigos": "coat",
#     "vestidos": "dress",
#     "monos": "jumpsuit",
#     "gafas de sol": "glasses",
#     "sombreros": "hat",
#     "corbatas": "tie",
#     "guantes y manopla": "glove",
#     "relojes": "watch",
#     "cinturones": "belt",
#     "calcetines y medias": "tights, stockings",
#     "bolso": "bag, wallet",  # Assuming both carteras and bolsos go here
#     "bufandas": "scarf",
#     "cazadoras": "jacket",
#     "chaquetas": "jacket",
#     "pulseras": "bracelet",
#     "collares": "necklace",
#     "trajes de baño": "swimsuit",
#     "ropa interior": "underwear",
#     "mochilas": "backpack"
# }

genre_map = {'mujer': 'woman', None: None, 'chica': 'girl', 'bebe': 'baby',
             'hombre': 'man', 'chico': 'boy'}

if __name__ == "__main__":
    start_id = 0

    # Directory paths
    input_directory = '~/Downloads/clothing/data'

    output_directory = '~/Downloads/clothing/adjusted-jsons'

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Process each file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.json'):
            with open(os.path.join(input_directory, filename), 'r') as file:
                data = json.load(file)

            # Prepare transformed data
            bbox_id = []
            category = []
            bbox = []
            area = []
            genre_list = []

            for i, box in enumerate(data['arr_boxes']):
                eng_category = spanish_to_english.get(box['class'], "undefined")
                if eng_category == "undefined":
                    print(box['class'])

                if eng_category != "undefined":
                    category_index = eng_category  # categories  # .index(eng_category)

                    # Calculate bbox format and area
                    x1, y1 = int(box['x']), int(box['y'])
                    x2, y2 = x1 + int(box['width']), y1 + int(box['height'])
                    bbox_format = [x1, y1, x2, y2]
                    box_area = (x2 - x1) * (y2 - y1)

                    # Append results to lists
                    bbox_id.append(start_id)
                    category.append(category_index)
                    bbox.append(bbox_format)
                    area.append(box_area)
                    genre_list.append(genre_map[box.get("genre")])
                    start_id += 1

            # Prepare output JSON
            output_json = {
                "bbox_id": bbox_id,
                "category": category,
                "bbox": bbox,
                "area": area,
                "genre": genre_list,
            }

            # Write output to a new file in the output directory
            output_file_path = os.path.join(output_directory, f'{filename}')
            with open(output_file_path, 'w') as output_file:
                json.dump(output_json, output_file, indent=4)

    print("All files processed and saved to the output directory.")
