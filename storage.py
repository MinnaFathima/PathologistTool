
import sqlite3
import json
import base64
from io import BytesIO
from PIL import Image

DB = "annotations.db"

create_sql = '''
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name TEXT,
    image_blob BLOB,
    heatmap_blob BLOB,
    annotations_json TEXT,
    final_label INTEGER,
    confidence REAL,
    created_at TEXT DEFAULT (datetime('now'))
)
'''

def init_db(db_path=DB):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(create_sql)
    conn.commit()
    conn.close()


def pil_to_bytes(img_pil, fmt='PNG'):
    buff = BytesIO()
    img_pil.save(buff, format=fmt)
    return buff.getvalue()


def save_annotation(image_name, pil_img, pil_heatmap, annotations, final_label, confidence, db_path=DB):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    img_blob = pil_to_bytes(pil_img)
    heat_blob = pil_to_bytes(pil_heatmap)
    annotations_json = json.dumps(annotations)
    cur.execute('''INSERT INTO annotations (image_name, image_blob, heatmap_blob, annotations_json, final_label, confidence)
                VALUES (?, ?, ?, ?, ?, ?)''', (image_name, img_blob, heat_blob, annotations_json, final_label, confidence))
    conn.commit()
    conn.close()


def export_annotations_csv(csv_path='annotations_export.csv', db_path=DB):
    import pandas as pd
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query('SELECT id, image_name, annotations_json, final_label, confidence, created_at FROM annotations', conn)
    df.to_csv(csv_path, index=False)
    conn.close()
    return csv_path