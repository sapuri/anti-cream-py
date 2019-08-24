import argparse
import os
import sys

import cv2
from google.cloud import automl_v1beta1
from google.protobuf.json_format import MessageToDict


def main():
    parser = argparse.ArgumentParser(description='AntiCreamPy: Censoring Vagina with Deep Neural Networks')
    parser.add_argument('input', help='Input your original image file path. (e.g. input.jpg)')
    parser.add_argument('-o', '--output', default='censored.jpg',
                        help='Input the output file path. (default: censored.jpg)')

    args = parser.parse_args()
    file_path = args.input
    output_file_path = args.output

    project_id = os.getenv("PROJECT_ID")
    model_id = os.getenv("MODEL_ID")

    try:
        prediction = MessageToDict(get_prediction(file_path, project_id, model_id))
    except BaseException as e:
        print('failed to get prediction:', e)
        sys.exit(1)

    img = cv2.imread(file_path)
    for payload in prediction['payload']:
        vertices = payload['imageObjectDetection']['boundingBox']['normalizedVertices']
        try:
            x1, y1, x2, y2 = convert_vertex(img, vertices)
        except BaseException as e:
            print('failed to convert vertex:', e)
            sys.exit(1)

        try:
            img = censor(img, x1, y1, x2, y2)
        except BaseException as e:
            print('failed to censor:', e)
            sys.exit(1)

    try:
        save_image(img, output_file_path)
    except BaseException as e:
        print('failed to save image:', e)
        sys.exit(1)

    print('Saved:', output_file_path)


def get_prediction(file_path: str, project_id: str, model_id: str) -> any:
    prediction_client = automl_v1beta1.PredictionServiceClient()

    # 'content' is base-64-encoded image data.
    with open(file_path, 'rb') as ff:
        content = ff.read()

    name = f'projects/{project_id}/locations/us-central1/models/{model_id}'
    payload = {'image': {'image_bytes': content}}
    params = {}
    return prediction_client.predict(name, payload, params)


def convert_vertex(img: any, vertices: list) -> tuple:
    h, w, _ = img.shape
    x1 = int(w * vertices[0]['x'])
    y1 = int(h * vertices[0]['y'])
    x2 = int(w * vertices[1]['x'])
    y2 = int(h * vertices[1]['y'])
    return x1, y1, x2, y2


def censor(img: any, x1: int, y1: int, x2: int, y2: int) -> any:
    vagina = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = mosaic(vagina)
    return img


def mosaic(img: any, ratio: int = 0.1) -> any:
    censored = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(censored, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)


def save_image(img: any, file_path: str):
    cv2.imwrite(file_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])


if __name__ == '__main__':
    main()
