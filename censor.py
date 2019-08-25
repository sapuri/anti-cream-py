import argparse
import os
import sys
from pathlib import Path

import cv2
from google.cloud import automl_v1beta1
from google.protobuf.json_format import MessageToDict


class Censor:
    def __init__(self, path: str, project_id: str, model_id: str):
        self.path = path
        self.project_id = project_id
        self.model_id = model_id

    def run(self):
        path = Path(self.path)
        if not path.exists():
            print(f'No such file or directory: \'{path}\'')
            sys.exit(1)

        if path.is_file():
            output_file_path = path.with_name(f'{path.stem}_censored{path.suffix}')
            self.process(str(path), str(output_file_path))
            return

        for file in path.iterdir():
            output_file_path = file.with_name(f'{file.stem}_censored{file.suffix}')
            self.process(str(file), str(output_file_path))

    def process(self, file_path: str, output_file_path: str):
        try:
            prediction = MessageToDict(self.get_prediction(file_path, self.project_id, self.model_id))
        except BaseException as e:
            print('Failed to get prediction:', e)
            sys.exit(1)

        img = cv2.imread(file_path)
        for payload in prediction['payload']:
            vertices = payload['imageObjectDetection']['boundingBox']['normalizedVertices']
            try:
                x1, y1, x2, y2 = self.convert_vertex(img, vertices)
            except BaseException as e:
                print('Failed to convert vertex:', e)
                sys.exit(1)

            try:
                img = self.mosaic(img, x1, y1, x2, y2)
            except BaseException as e:
                print('Failed to censor:', e)
                sys.exit(1)

        try:
            self.save_image(img, output_file_path)
        except BaseException as e:
            print('Failed to save image:', e)
            sys.exit(1)

        print('Saved:', output_file_path.split('/')[-1])

    @staticmethod
    def get_prediction(file_path: str, project_id: str, model_id: str):
        prediction_client = automl_v1beta1.PredictionServiceClient()

        # 'content' is base-64-encoded image data.
        with open(file_path, 'rb') as ff:
            content = ff.read()

        name = f'projects/{project_id}/locations/us-central1/models/{model_id}'
        payload = {'image': {'image_bytes': content}}
        params = {}
        return prediction_client.predict(name, payload, params)

    @staticmethod
    def convert_vertex(img: any, vertices: list) -> tuple:
        h, w, _ = img.shape
        x1 = int(w * vertices[0]['x'])
        y1 = int(h * vertices[0]['y'])
        x2 = int(w * vertices[1]['x'])
        y2 = int(h * vertices[1]['y'])
        return x1, y1, x2, y2

    @staticmethod
    def mosaic(img: any, x1: int, y1: int, x2: int, y2: int, ratio: float = 0.1) -> any:
        vagina = img[y1:y2, x1:x2]
        censored = cv2.resize(cv2.GaussianBlur(vagina, (25, 25), 10), None, fx=ratio, fy=ratio,
                              interpolation=cv2.INTER_NEAREST)
        img[y1:y2, x1:x2] = cv2.resize(censored, vagina.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        return img

    @staticmethod
    def save_image(img: any, file_path: str):
        cv2.imwrite(file_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AntiCreamPy: Censoring Vagina with Deep Neural Networks')
    parser.add_argument('input',
                        help='Input your original image file path. (e.g. input.jpg) If a directory is specified, all image files in that directory will be processed. (e.g. ./images)')

    args = parser.parse_args()
    path = args.input

    project_id = os.getenv("PROJECT_ID")
    model_id = os.getenv("MODEL_ID")

    censor = Censor(path, project_id, model_id)
    censor.run()
