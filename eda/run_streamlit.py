from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import json
import glob
import torch

IMAGES_PER_PAGE = 16
ROOT_PATH = {
    'Train' : '../data/train', 
    'Test' : '../data/test',
    'Inferred Train' : '../data/train'
}
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

def get_label(image_path):
    image = cv2.imread(image_path)
    image = image / 255.

    label_path = image_path.replace('DCM', 'outputs_json').replace('.png', '.json')
    label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
    label = np.zeros(label_shape, dtype=np.uint8)

    with open(label_path, "r") as f:
        annotations = json.load(f)
    annotations = annotations["annotations"]

    for ann in annotations:
        c = ann["label"]
        class_ind = CLASS2IND[c]
        points = np.array(ann["points"])

        class_label = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(class_label, [points], 1)
        label[..., class_ind] = class_label

    label = label.transpose(2, 0, 1)
    label = torch.from_numpy(label).float()

    return label


def get_prediction(rles):
    preds = []
    for rle in rles:
        pred = rle2mask(rle, height=2048, width=2048)
        preds.append(pred)

    if preds:
        return np.stack(preds, 0)
    else:
        return np.zeros((1, 256, 256), dtype=np.uint8)


def label2rgb(label, selected_classes=None):
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)

    for i, class_label in enumerate(label):
        if selected_classes is None or CLASSES[i] in selected_classes:
            image[class_label == 1] = PALETTE[i]

    return image


def rle2mask(rle, height, width): 
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)


def display_images(images, segmentations, image_paths):
    """
    num_rows x 4 형태로 이미지 생성 
    """
    num_rows = IMAGES_PER_PAGE // 4 
    for i in range(num_rows):
        cols = st.columns(4)
        for j in range(4):
            idx = (i * 2) + (j // 2)
            with cols[j]:
                if j % 2 == 0:
                    st.image(images[idx], caption=image_paths[idx], use_container_width=True)
                else:
                    st.image(segmentations[idx], use_container_width=True)  


def load_image_paths(dataset_option):
    image_paths = sorted(glob.glob(ROOT_PATH[dataset_option] + '/DCM/*/*.png'))
    json_paths = sorted(glob.glob(ROOT_PATH[dataset_option] + '/outputs_json/*/*.json'))
    assert len(image_paths) == len(json_paths), "len(image) & len(json) unmatched"
    return image_paths


def load_images(image_paths):
    return [Image.open(pth) for pth in image_paths]


@st.cache_data
def input_fname():
    fname = input("Enter the CSV filename: ")
    return fname


@st.cache_data
def load_csv(fname):
    return pd.read_csv(fname)


def update_pagination(total_pages):
    if "page" not in st.session_state:
        st.session_state.page = 0

    page_numbers = list(range(total_pages))
    st.sidebar.selectbox(
        "Go to page",
        options=page_numbers,
        index=st.session_state.page,
        key="page_select",
        on_change=lambda: st.session_state.update({"page": st.session_state.page_select})
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button('Prev') and st.session_state.page > 0:
            st.session_state.page -= 1
    with col2:
        if st.sidebar.button('Next') and st.session_state.page < total_pages - 1:
            st.session_state.page += 1


def get_paged_data(image_paths):
    total_pages = len(image_paths) // IMAGES_PER_PAGE
    update_pagination(total_pages)
    start_idx = st.session_state.page * IMAGES_PER_PAGE
    end_idx = start_idx + IMAGES_PER_PAGE // 2
    return image_paths[start_idx:end_idx]


def load_ground_truth_labels(image_paths, selected_classes):
    gt = []
    for pth in image_paths:
        label = get_label(pth)
        gt.append(label2rgb(label, selected_classes))
    return gt


def load_predictions(image_paths, df, selected_classes):
    preds = []
    for pth in image_paths:
        image_name = pth.split('/')[-1]
        image_df = df[df['image_name'] == image_name]
        rles = image_df['rle'].tolist()
        pred = get_prediction(rles)
        preds.append(label2rgb(pred, selected_classes))
    return preds


def display_legend():
    with st.popover("Classes Legend"):
        cols = st.columns(4)
        for i, (class_name, color) in enumerate(zip(CLASSES, PALETTE)):
            with cols[i % 4]:
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.image(Image.new('RGB', (20, 20), color), width=30)
                with col2:
                    st.write(class_name)


def main():
    st.set_page_config(page_title='Visualization', layout='wide')
    if st.sidebar.button("Refresh"):
        st.rerun()
    st.title("Hand Bone Image Segmentation")

    dataset_option = st.sidebar.selectbox(
        "Choose Train / Inferred Train / Test Dataset",
        ("Train", "Inferred Train", "Test")
    )

    class_option = st.sidebar.pills(
        "Choose handbone class. (Whole handbones will be shown if not selected)",
        options=CLASSES,
        selection_mode='multi',
        default=None
    )

    selected_classes = CLASSES if not class_option else class_option

    if dataset_option == 'Train':
        st.header("Train data & Ground Truth")
        display_legend()
        image_paths = load_image_paths(dataset_option)
        image_paths = get_paged_data(image_paths)
        images = load_images(image_paths)
        gt = load_ground_truth_labels(image_paths, selected_classes)
        display_images(images, gt, image_paths)

    elif dataset_option == 'Test':
        st.header("Test data & Inference")
        display_legend()
        csv_fname = input_fname()
        df = load_csv(csv_fname)
        image_paths = sorted(glob.glob(ROOT_PATH[dataset_option] + '/DCM/*/*.png'))
        image_paths = get_paged_data(image_paths)
        images = load_images(image_paths)
        preds = load_predictions(image_paths, df, selected_classes)
        display_images(images, preds, image_paths)

    elif dataset_option == 'Inferred Train':
        st.header("Ground Truth & Inference")
        display_legend()
        csv_fname = input_fname()
        df = load_csv(csv_fname)
        image_paths = sorted(glob.glob(ROOT_PATH[dataset_option] + '/DCM/*/*.png'))
        image_paths = get_paged_data(image_paths)
        images = load_images(image_paths)
        gt = load_ground_truth_labels(image_paths, selected_classes)
        preds = load_predictions(image_paths, df, selected_classes)
        display_images(gt, preds, image_paths)

        
if __name__ == "__main__":
    main()
