import datetime
import zipfile
import json
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.backend import clear_session
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History
from tqdm import tqdm
from pathlib import Path

from model_analyzer import ModelAnalyzer
from scoring import *
from inference.predict_image import *
from postprocessing import smooth_tiled_prediction

from datasets.dd_dataset_config import test_ids


class Experiment:

    def __init__(self, title, dataset, model_backend, batch_size, experiment_directory="", load_best=False,
                 enable_tensorboard=False):
        # NEW EXPERIMENT
        if experiment_directory == "":
            self.experiment_title = f"{title}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            self.basedir = os.path.join(f"{os.getcwd()}/experiments", self.experiment_title)
            self.init_experiment_directory_structure()
            self.model_backend = model_backend.compile()
        # LOAD EXPERIMENT
        else:
            self.experiment_title = experiment_directory
            self.basedir = os.path.join(f"{os.getcwd()}/experiments", experiment_directory)
            print(f"Loading experiment {experiment_directory}")
            if load_best:
                self.model_backend = model_backend.load(f"{self.basedir}/models/checkpoint")
            else:
                self.model_backend = model_backend.load(f"{self.basedir}/models/last_epoch_model")
        self.tensorboard_log = f"{self.basedir}/tensorboard_logs"
        self.dataset = dataset
        self.batch_size = batch_size
        self.model_analyzer = ModelAnalyzer(self.model_backend)
        self.scoring_backend = ChipSetScoring(self.basedir)
        self.enable_tensorboards = enable_tensorboard

    def init_experiment_directory_structure(self):
        os.makedirs(self.basedir)
        os.makedirs(f"{self.basedir}/models")
        os.makedirs(f"{self.basedir}/predictions")
        os.makedirs(f"{self.basedir}/predictions/chips")
        os.makedirs(f"{self.basedir}/export")

    def save_config(self):
        pass

    def analyze(self):
        self.model_analyzer.analyze(self.batch_size)
        self.model_analyzer.saveToJSON(self.basedir)

    def train(self, epochs):
        train_data, valid_data = self.dataset.load_dataset(self.batch_size)
        history = History()
        callbacks = [ModelCheckpoint(
            filepath=f"{self.basedir}/models/checkpoint",
            save_weights_only=True,
            monitor='val_mIOU',
            mode='max',
            save_best_only=True
        ),
            history,
        ]
        if self.enable_tensorboards:
            callbacks.append(TensorBoard(
                log_dir=self.tensorboard_log,
                histogram_freq=1,
                write_images=True,
                update_freq='epoch',
                profile_batch='500,510',
                embeddings_freq=1,
            ))
        self.model_backend.fit(
            train_data,
            validation_data=valid_data,
            epochs=epochs,
            callbacks=callbacks
        )
        self.plot_segm_history(history)
        with open(f"{self.basedir}/train_history.json", 'w') as outfile:
            json.dump(history.history, outfile)

    def predict(self, image_file, output, postprocessing=False, save_overlay=False):
        if postprocessing:
            smooth_tiled_prediction(self.model_backend, self.dataset.chip_size, 6, image_file, output)
        else:
            generate_predict_image(self.basedir, image_file, output, self.model_backend, self.dataset.chip_size, save_overlay=True)
        return os.path.join(self.basedir, f'predictions/{output}')

    def score(self):
        scores = self.scoring_backend.score_predictions(self.dataset.dataset_name)
        with open(f"{self.basedir}/scores.json", 'w') as score_json:
            json.dump(scores, score_json)

    def score_generalization(self, gsd=10):
        if gsd == 10:
            path = "./dataset-potsdam/2_Ortho_RGB_gsd10"
        else:
            path = "./dataset-potsdam/2_Ortho_RGB"
        potsdam_images = os.listdir(path)
        if not os.path.isdir(f"{self.basedir}/predictions/potsdam"):
            os.makedirs(f"{self.basedir}/predictions/potsdam")
            print("Creating Potsdam image predictions")
            for image in tqdm(potsdam_images):
                generate_predict_image(self.basedir, f"{path}/{image}", f"potsdam/{image[:-4]}",
                                       self.model_backend, self.dataset.chip_size)
        self.scoring_backend = PotsdamScoring(self.basedir, gsd)
        scores = self.scoring_backend.score_predictions("dataset-potsdam")
        with open(f"{self.basedir}/potsdam_scores.json", 'w') as score_json:
            json.dump(scores, score_json)

    def benchmark_inference(self):
        print("Starting Benchmark")
        chip_file_list = [f"./{self.dataset.dataset_name}/image-chips/test/{chip_file}"
                          for chip_file in os.listdir(f"./{self.dataset.dataset_name}/image-chips/test")]
        inference_timings = predict_chips_benchmark(self.basedir, chip_file_list, self.model_backend)
        with open(f"{self.basedir}/inferenc_benchmark.json", "w") as inference_json:
            json.dump({"timings": inference_timings,
                       "mean": np.mean(inference_timings),
                       "std": np.std(inference_timings),
                       "median": np.median(inference_timings),
                       "90_perc": np.percentile(inference_timings, 90)}, inference_json)

    def export_model(self):
        Path("./onnx_export").mkdir(parents=True, exist_ok=True)
        self.model_backend.save(f"{self.basedir}/export/{self.experiment_title}")
        os.system(f"python -m tf2onnx.convert --saved-model {self.basedir}/export/{self.experiment_title} --opset 12 --output ./onnx_export/{self.experiment_title}")

    def plot_segm_history(self, history, metrics=["mIOU", "val_mIOU"], losses=["loss", "val_loss"]):
        """[summary]
        https://github.com/karolzak/keras-unet
        Args:
            history ([type]): [description]
            metrics (list, optional): [description]. Defaults to ["miou", "val_miou"].
            losses (list, optional): [description]. Defaults to ["loss", "val_loss"].
        """
        # summarize history for iou
        plt.figure(figsize=(12, 6))
        for metric in metrics:
            plt.plot(history.history[metric], linewidth=3)
        plt.suptitle("metrics over epochs", fontsize=20)
        plt.ylabel("mIOU", fontsize=20)
        plt.xlabel("epoch", fontsize=20)
        plt.legend(metrics, loc="center right", fontsize=15)
        plt.savefig(f"{self.basedir}/plotted_mIOU.png")
        # summarize history for loss
        plt.figure(figsize=(12, 6))
        for loss in losses:
            plt.plot(history.history[loss], linewidth=3)
        plt.suptitle("loss over epochs", fontsize=20)
        plt.ylabel("loss", fontsize=20)
        plt.xlabel("epoch", fontsize=20)
        plt.legend(losses, loc="center right", fontsize=15)
        plt.savefig(f"{self.basedir}/plotted_loss.png")

    # Saving model weights of the last epoch
    # Best mIOU related epoch is saved in the checkpoint file
    def save_model(self):
        print("Saving last epoch model")
        self.model_backend.save_weights(f"{self.basedir}/models/last_epoch_model")

    # creating zip directory of experiment models
    def bundle(self):
        print("Creating zip bundle")
        compression = zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(f"{self.basedir}/{self.experiment_title}.zip", mode='w') as zip_file:
            zip_directory(zip_file, "models", compression, f"{self.basedir}/models")
            if self.enable_tensorboards:
                zip_directory(zip_file, "tensorboard_logs", compression, f"{self.basedir}/tensorboard_logs")
            zip_file.write(f"{self.basedir}/model_summary.json", "model_summary.json", compress_type=compression)
            zip_file.write(f"{self.basedir}/plotted_loss.png", "plotted_loss.png", compress_type=compression)
            zip_file.write(f"{self.basedir}/plotted_mIOU.png", "plotted_mIOU.png", compress_type=compression)
