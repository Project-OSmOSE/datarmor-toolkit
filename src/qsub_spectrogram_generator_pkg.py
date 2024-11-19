from OSmOSE import Spectrogram
from OSmOSE.config import *
import argparse
import random
import os
import numpy as np
import itertools
import pandas as pd
from pathlib import Path
from OSmOSE.utils.audio_utils import get_all_audio_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Python script to process a list of file and generate spectrograms."
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--dataset-sr", "-s", required=True, help="The analysis frequency."
    )
    required.add_argument(
        "--dataset-path", "-p", required=True, help="The path to the dataset folder"
    )
    parser.add_argument("--files", "-f", nargs="*")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deletes all existing spectrograms and their zoom levels matching the audio file before processing. If some spectrograms do not match any processed audio files, they will not be deleted.",
    )
    parser.add_argument(
        "--batch-ind-min",
        "-min",
        type=int,
        default=0,
        help="The first file to consider. Default is 0.",
    )
    parser.add_argument(
        "--batch-ind-max",
        "-max",
        type=int,
        default=-1,
        help="The last file to consider. -1 means consider all files from batch-ind-min. Default is -1",
    )
    parser.add_argument(
        "--save-matrix",
        "-m",
        action="store_true",
        help="Whether to save the spectrogram matrices or not. Note that activating this parameter might increase greatly the volume of the project.",
    )
    parser.add_argument("--save-for-LTAS", action="store_true")
    parser.add_argument("--spectrogram-metadata-path", type = str, default = '')

    args = parser.parse_args()

    print("Parameters :", args)

    os.system("ln -sf /appli/sox/sox-14.4.2_gcc-7.2.0/bin/sox sox")

    dataset = Spectrogram.from_csv(dataset_path = args.dataset_path, metadata_csv_path = args.spectrogram_metadata_path)

    if not dataset.path.joinpath("processed", "spectrogram", "adjust_metadata.csv"):
        raise FileNotFoundError(
            f"The file adjust_metadata.csv has not been found in the processed/spectrogram folder. Consider using the initialize() or update_parameters() methods."
        )

    files = get_all_audio_files(dataset.audio_path)

    if args.files:
        selected_files = args.files
        if not all(dataset.audio_path.joinpath(f) in files for f in selected_files):
            raise FileNotFoundError(
                f"At least one file in {selected_files} has not been found in {files}"
            )
        else:
            files = selected_files

    print(f"Found {len(files)} files in {dataset.audio_path}.")

    files_to_process = files[
        args.batch_ind_min : (
            args.batch_ind_max + 1 if args.batch_ind_max != -1 else len(files)
        )
    ]

    print(f"files to process: {files_to_process}\n")

    for i, audio_file in enumerate(files_to_process):
        print(audio_file)

        dataset.process_file(
            audio_file,
            adjust=False,
            save_matrix=args.save_matrix,
            save_for_LTAS=args.save_for_LTAS,
            clean_adjust_folder=True,
            overwrite=args.overwrite,
        )

    if args.save_for_LTAS and args.save_matrix:

        # get metadata from spectrogram folder
        metadata_path = next(
            dataset.path.joinpath(
                OSMOSE_PATH.spectrogram, dataset.audio_path.name
            ).rglob("metadata.csv"),
            None,
        )
        metadata_spectrogram = pd.read_csv(metadata_path)

        path_all_welch = dataset.path.joinpath(
            OSMOSE_PATH.welch,
            dataset.audio_path.name,
            f"batch_{args.batch_ind_min}_welch.npz",
        )
        Sxx = np.empty((1, int(metadata_spectrogram["nfft"][0] / 2) + 1))
        Time = []

        print(f"number of welch: {len(files_to_process)}")

        for file_npz in files_to_process:
            file_npz = dataset.path.joinpath(
                OSMOSE_PATH.welch,
                dataset.audio_path.name,
                file_npz.with_suffix(".npz").name,
            )
            if file_npz.exists():
                print(f"load {file_npz}")
                current_matrix = np.load(file_npz, allow_pickle=True)
                os.remove(file_npz)
                Sxx = np.vstack((Sxx, current_matrix["Sxx"]))
                print(f"Sxx {Sxx.shape}")
                Time.append(current_matrix["Time"])
                print(f"time {len(Time)}")
            else:
                print(f"File {file_npz} not found. Skipping...")

        Sxx = Sxx[1:, :]
        Freq = current_matrix["Freq"]

        Time = np.array(Time)

        # flatten time, which is currently a list of arrays
        if Time.ndim == 2:
            Time = list(itertools.chain(*Time))
        else:
            Time = [
                tt.item() for tt in Time
            ]  # suprinsingly , doing simply = list(time) was droping the Timestamp dtype, to be investigated in more depth...

        np.savez(path_all_welch, Sxx=Sxx, Time=Time, Freq=Freq, allow_pickle=True)
