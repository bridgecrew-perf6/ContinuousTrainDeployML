import pickle
import sys
import torch
import uvicorn
import yaml
import utils.inference as inference 
import utils.image_utils as image_utils
import utils.postprocessing as postpro
from tempfile import NamedTemporaryFile
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from pathlib import Path


# TODO this should probably be defined somewhere, where the training can also access it?!
print('loading configuration...', end='')
with open("config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
print('done')

metadata_path = Path(cfg['deploy']['metadata_path']) 

with open(metadata_path / cfg['deploy']['label_map'] , "rb") as f:
    id2label = pickle.load(f)
    print('loaded label map')

tokenizer = inference.load_tokenizer(pretrained_model=cfg['general']['processor'])
print('Tokenizer loaded')

device = torch.device('cpu')
print(f'Loading model using device: {device}...', end='')
model = inference.load_model(metadata_path / cfg['deploy']['model_deployed'], device)

target_size =  tuple(cfg['preprocessing']['target_size'])

app = Starlette()


@app.route('/')
def root_route(request: Request):
    return PlainTextResponse('Text Extraction Server')


@app.route('/classify', methods=['POST', 'PUT'])
async def classify(request: Request):

    content_type = request.headers.get('content-type')

    if content_type is None:
        return PlainTextResponse(f'Content-Type header must be set. Only images and PDFs are supported.',
                                 status_code=415)
    if not content_type.startswith('image/') and not content_type.startswith('application/pdf'):
        return PlainTextResponse(f'Unsupported content type "{content_type}". Only images and PDFs are supported.',
                                 status_code=415)

    body = await request.body()

    suffix = '.jpg'
    if content_type.startswith('image/png'):
        suffix = '.png'
    elif content_type.startswith('application/pdf'):
        suffix = '.pdf'

    with NamedTemporaryFile(suffix=suffix) as tf:
        tf.write(body)
        tf.flush()

        img_list = image_utils.convert_resize_documents(tf.name, target_size=target_size)
        # takes first page only. Extend?
        image = img_list[0]   

        bboxes, words = inference.ocr_processing(image, target_size=target_size)

        encoded_list, wids = inference.encode_data(image, words, bboxes, tokenizer, device=device)

        extractions = list()
        for encoded_inputs, word_ids in zip(encoded_list, wids):
            predictions = inference.inference(model, encoded_inputs)
            text_extraction = inference.map_output(predictions, label_map=id2label, word_ids=word_ids, ocr_words=words)
            extractions.append(text_extraction)

        key_info = postpro.output_postprocessing(extractions)

        #predictions = inference(model, encoded_inputs)
        #text_extraction = map_output(predictions, label_map=id2label, word_ids=encoded_inputs.word_ids(0), ocr_words=words)

        return JSONResponse({
            'key_information': key_info
        })

@app.route('/classify_extended_output', methods=['POST', 'PUT'])
async def classify_extended_output(request: Request):

    content_type = request.headers.get('content-type')

    if content_type is None:
        return PlainTextResponse(f'Content-Type header must be set. Only images and PDFs are supported.',
                                 status_code=415)
    if not content_type.startswith('image/') and not content_type.startswith('application/pdf'):
        return PlainTextResponse(f'Unsupported content type "{content_type}". Only images and PDFs are supported.',
                                 status_code=415)

    body = await request.body()

    suffix = '.jpg'
    if content_type.startswith('image/png'):
        suffix = '.png'
    elif content_type.startswith('application/pdf'):
        suffix = '.pdf'

    with NamedTemporaryFile(suffix=suffix) as tf:
        tf.write(body)
        tf.flush()

        img_list = image_utils.convert_resize_documents(tf.name, target_size=target_size)
        # takes first page only. Extend?
        image = img_list[0]   

        bboxes, words = inference.ocr_processing(image, target_size=target_size)

        encoded_inputs = inference.encode_data(image, words, bboxes, processor, device=device)

        predictions = inference.inference(model, encoded_inputs)

        word_labelids = inference.map_predictions_to_words(predictions, encoded_inputs.word_ids(0))

        unlabeled_words = len(words) - len(word_labelids)
        word_labelids += [0] * unlabeled_words

        result = [{'bbox': b, 'word': w, 'prediction': id2label[l]} for b, w, l in zip(bboxes, words, word_labelids)]
            
        return JSONResponse(result)


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
