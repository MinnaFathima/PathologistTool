import streamlit as st
from PIL import Image
import io
from inference import InferenceEngine
from utils import overlay_heatmap_on_image
from storage import init_db, save_annotation, export_annotations_csv
import tempfile
import os
import json

st.set_page_config(layout='wide', page_title='Interactive Pathologist Tool')

st.title('Interactive Pathologist Tool — Prototype')

# Sidebar: model path and init
st.sidebar.header('Model / Settings')
model_checkpoint = st.sidebar.text_input('Model checkpoint path (local)', 'model_checkpoint.pth')
use_gpu = st.sidebar.checkbox('Use GPU if available', value=True)
init_db()  # ensure DB exists

# Lazy init inference engine
@st.experimental_singleton
def get_engine(path, use_gpu):
    device = None
    if use_gpu:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from inference import InferenceEngine
    engine = InferenceEngine(model_checkpoint=path, device=device)
    return engine

if not os.path.exists(model_checkpoint):
    st.sidebar.error('Please provide a valid model checkpoint path on the machine running this app.')

engine = None
if os.path.exists(model_checkpoint):
    try:
        engine = get_engine(model_checkpoint, use_gpu)
    except Exception as e:
        st.sidebar.error(f'Failed to load model: {e}')

# Main UI
uploaded = st.file_uploader('Upload a microscopic image (jpg/png/tif)', type=['png','jpg','jpeg','tif','tiff'])

if uploaded is not None and engine is not None:
    # read image
    image = Image.open(uploaded).convert('RGB')
    st.image(image, caption='Uploaded image', use_column_width=True)

    with st.spinner('Running inference...'):
        heatmap, confidence = engine.predict_whole_image(image)
        overlay = overlay_heatmap_on_image(image, heatmap, alpha=0.5)

    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader('AI heatmap overlay')
        st.image(overlay, use_column_width=True)
    with col2:
        st.subheader('Controls & Annotations')
        st.write(f'Predicted tumor confidence (avg over tiles): **{confidence:.3f}**')

        # annotation input
        st.write('Add final decision:')
        final_label = st.radio('Final label', options=[0,1], format_func=lambda x: 'Normal' if x==0 else 'Tumor')

        # annotation drawing: simple JSON boxes list to store
        st.write('Add rectangular annotations (optional). Provide as JSON list:')
        default_json = '[]'
        ann_json = st.text_area('Annotations JSON', value=default_json, height=120)
        try:
            ann_data = json.loads(ann_json)
        except Exception:
            ann_data = []
            st.error('Invalid JSON for annotations')

        if st.button('Save annotation & AI outputs'):
            try:
                save_annotation(os.path.basename(uploaded.name), image, overlay, ann_data, int(final_label), float(confidence))
                st.success('Saved to database')
            except Exception as e:
                st.error(f'Failed to save: {e}')

        if st.button('Export annotations to CSV'):
            path = export_annotations_csv()
            st.success(f'Exported to {path}')

    st.markdown('---')
    st.info('This prototype stores images & annotations in a local SQLite DB named `annotations.db`. For production use, switch to MongoDB and secure storage.')