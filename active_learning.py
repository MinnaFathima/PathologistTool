# simple helper to extract tiles and labels from stored annotations for retraining
import sqlite3
import json
from PIL import Image
from io import BytesIO
import os
from utils import tile_image

DB = 'annotations.db'

def export_training_data(output_dir='al_data', tile_size=224, stride=224):
    os.makedirs(output_dir, exist_ok=True)
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute('SELECT id, image_blob, final_label FROM annotations')
    rows = cur.fetchall()
    for rid, img_blob, label in rows:
        img = Image.open(BytesIO(img_blob)).convert('RGB')
        tiles, coords = tile_image(img, tile_size=tile_size, stride=stride)
        for i, t in enumerate(tiles):
            fname = os.path.join(output_dir, f'{rid}_{i}_{label}.png')
            t.save(fname)
    conn.close()
    print('Exported training tiles to', output_dir)

if __name__ == '__main__':
    export_training_data()