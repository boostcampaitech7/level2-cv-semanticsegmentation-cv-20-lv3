import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from function import decode_rle_to_mask


def decode_rles_to_masks(rles, height, width):
    """
    여러 RLE 문자열을 한 번에 디코딩합니다. NaN 값은 빈 마스크로 처리합니다.
    
    Args:
        rles (list): RLE 문자열 리스트.
        height (int): 이미지 높이.
        width (int): 이미지 너비.

    Returns:
        list: 디코딩된 마스크 리스트 (numpy 배열 형태).
    """
    masks = []
    for idx, rle in tqdm(enumerate(rles), desc="Decoding RLEs", total=len(rles)):
        if pd.isna(rle):  # NaN 값 처리
            masks.append(np.zeros((height, width), dtype=np.uint8))
        else:
            masks.append(decode_rle_to_mask(rle, height, width))
    return masks


def csv_to_json_with_classes(result_file_name, output_dir, height=2048, width=2048):
    """
    NaN 값을 처리하며 CSV 데이터를 빠르게 JSON으로 변환합니다. class 정보도 포함됩니다.
    
    Args:
        result_file_name (str): CSV 파일 경로.
        output_dir (str): JSON 파일 저장 경로.
        height (int, optional): 이미지 높이. 기본값은 2048.
        width (int, optional): 이미지 너비. 기본값은 2048.
    """
    try:
        # CSV 파일 읽기
        db = pd.read_csv(result_file_name)
        print(f"CSV 파일이 성공적으로 로드되었습니다: {result_file_name}")

        # RLE 디코딩 및 포인트 변환
        masks = decode_rles_to_masks(db['rle'].values, height, width)
        points_list = [np.argwhere(mask == 1).tolist() for mask in tqdm(masks, desc="Converting Masks to Points")]

        # 이미지별 annotations 생성
        annotations = (
            db.assign(points=points_list)  # 포인트 리스트 추가
            .groupby('image_name')  # 이미지 이름별로 그룹화
            .apply(
                lambda group: [
                    {
                        "id": idx,
                        "type": "poly_seg",
                        "attributes": {"class": class_name},  # class 정보 추가
                        "points": points,
                        "label": class_name
                    }
                    for idx, (points, class_name) in zip(group.index, zip(group['points'], group['class']))
                ]
            )
        )

        # JSON 파일 저장
        os.makedirs(output_dir, exist_ok=True)
        total_files = len(annotations)
        with tqdm(total=total_files, desc="Saving JSON Files") as pbar:
            for image_name, annotation_list in annotations.items():
                json_file_name = os.path.join(output_dir, f"{image_name.replace('.png', '.json')}")
                with open(json_file_name, 'w') as json_file:
                    json.dump({"annotations": annotation_list}, json_file, indent=4)
                pbar.update(1)  # tqdm 업데이트
                pbar.set_postfix({"Current File": json_file_name})

    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {result_file_name}")
        raise e
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        raise e


# 사용 예시
csv_to_json_with_classes(
    result_file_name='/data/ephemeral/home/level2-cv-semanticsegmentation-cv-20-lv3/result/output/256.csv',
    output_dir='/data/ephemeral/home/level2-cv-semanticsegmentation-cv-20-lv3/result/jsons/',
    height=2048,
    width=2048
)