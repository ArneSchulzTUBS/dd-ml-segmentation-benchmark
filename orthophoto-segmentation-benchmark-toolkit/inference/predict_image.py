import numpy as np
from PIL import Image
from util import *
import cv2
from timeit import default_timer as timer
from tqdm import tqdm


def predict_chips_benchmark(basedir, chip_file_list, model, save_predictions=True):
    # LOAD CHIPS IN MEMORY
    print(f"Loading {len(chip_file_list)} chips")
    chip_files = [(np.array(Image.open(chip_file).convert('RGB')), os.path.basename(chip_file)) for chip_file in chip_file_list]
    inference_timings = []
    predictions = []

    print("Warmup")
    for i in range(50):
        chip, chip_id = chip_files[0]
        chip = np.expand_dims(np.array(chip), axis=0)
        model.predict(chip)

    for chip, chip_id in tqdm(chip_files):
        chip = np.expand_dims(np.array(chip), axis=0)
        start = timer()
        predicted_chip = model.predict(chip)
        end = timer()
        inference_timings.append(end - start)
        predictions.append((np.squeeze(predicted_chip), chip_id))
    print(f"Mean: {np.mean(inference_timings)}, STD: {np.std(inference_timings)}, MEDIAN: {np.median(inference_timings)}")
    if save_predictions:
        print("Saving prediction chips")
        if not os.path.isdir(f"{basedir}/predictions/test-chip-predictions"):
            os.mkdir(f"{basedir}/predictions/test-chip-predictions")
        predictions[0] = np.array(predictions[0])
        for prediction, chip_id in predictions:
            category_chip = np.argmax(prediction, axis=-1) + 1
            mask = category2mask(category_chip)
            mask_img = Image.fromarray(mask)
            mask_img.save(f"{basedir}/predictions/test-chip-predictions/{chip_id}")
    return inference_timings


def generate_predict_image(basedir, input_file, output, model, chip_size, save_prediction=True, save_overlay=False):
    output_file = os.path.join(basedir, f'predictions/{output}.png')
    size = chip_size
    with Image.open(input_file).convert('RGB') as img:
        nimg = np.array(Image.open(input_file).convert('RGB'))
        shape = nimg.shape
        chips = chips_from_image(nimg, chip_size)

        chips = [(chip, xi, yi) for chip, xi, yi in chips if chip.sum() > 0]
        prediction = np.zeros(shape[:2], dtype='uint8')

        chip_preds = []
        for chip in np.array([chip for chip, _, _ in chips]):
            np.expand_dims(chip, axis=0)
            predicted_chip = model.predict(np.expand_dims(np.array(chip), axis=0))
            chip_preds.append(np.squeeze(predicted_chip))
        chip_preds = np.array(chip_preds)

        for (chip, x, y), pred in zip(chips, chip_preds):
            category_chip = np.argmax(pred, axis=-1) + 1
            section = prediction[y:y + size, x:x + size].shape
            prediction[y:y + size, x:x + size] = category_chip[:section[0], :section[1]]

        mask = category2mask(prediction)
        mask_img = Image.fromarray(mask)
        if save_prediction:
            mask_img.save(output_file)
            if save_overlay:
                prediction_overlay_image = Image.blend(img, mask_img, alpha=0.5)
                prediction_overlay_image.save(f"{output_file[:-4]}-overlay.png")
        else:
            return mask
        # generate_prediction_chips(np.array(prediction_overlay_image), chip_size, basedir, output)


def generate_prediction_chips(prediction_overlay_image, chip_size, basedir, scene):
    if not os.path.isdir(f"{basedir}/predictions/chips"):
        os.makedirs(f"{basedir}/predictions/chips")
    chips = chips_from_image(prediction_overlay_image, chip_size)
    for i, chip in enumerate(chips):
        Image.fromarray(chip[0]).save(f"{basedir}/predictions/chips/{scene}-{i}.png")
