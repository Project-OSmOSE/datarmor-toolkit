import argparse
import os
import numpy as np
import itertools
import pandas as pd

from OSmOSE import Spectrogram
from pathlib import Path
from OSmOSE.config import OSMOSE_PATH

def _get_files_processed_by_current_batch(all_files: list[Path], batch_index: int, total_number_of_batches: int) -> list[Path]:
    """
    Compute which files are concerned by this batch.

    Parameters
    ----------
    all_files: list[str]
        List of all the files processed by all batches
    batch_index:
        Index of the current batch
    total_number_of_batches:
        Total number of batches running the process


    Returns
    -------
    list[pathlib.Path]:
        List of the files processed by the current batch
    """

    batch_sizes = _compute_batch_sizes(len(all_files), total_number_of_batches)
    batch_first_file_indexes = [sum(batch_sizes[:i]) for i in range(len(batch_sizes))]
    first_file_index = batch_first_file_indexes[batch_index]
    last_file_index = first_file_index + batch_sizes[batch_index]
    return all_files[first_file_index:last_file_index]

def _get_all_processed_files(datetime_begin: pd.Timestamp, datetime_end: pd.Timestamp, audio_folder: Path) -> list[Path]:
    """
    Compute which files are concerned by the whole analysis (all batches included).

    Since the timestamp.csv file might not exist at the time where this job file is created,
    every batch has to compute it (since the execution of this file is delayed until all
    initialize() jobs are completed).
    Parameters
    ----------
    datetime_begin: pandas.Timestamp
        Timestamp of the beginning of the process
    datetime_end: pandas.Timestamp
        Timestamp of the end of the process
    audio_folder: pathlib.Path
        Path to the reshaped audio folder

    Returns
    -------
    list[pathlib.Path]:
        The list of files for which spectrograms should be plotted.
    """
    timestamps = pd.read_csv(audio_folder / "timestamp.csv")
    file_duration = float(pd.read_csv(audio_folder / "metadata.csv")["audio_file_dataset_duration"][0])
    timestamps["begin"] = timestamps["timestamp"].apply(lambda t: pd.Timestamp(t))
    timestamps["end"] = timestamps["begin"] + pd.Timedelta(seconds=file_duration)
    return [Path(audio_folder/filename) for filename in timestamps.loc[(datetime_begin < timestamps["end"]) & (datetime_end > timestamps["begin"]), "filename"]]

def _compute_batch_sizes(nb_files: int, nb_batches: int) -> list[int]:
    """
    Compute the number of files processed by each batch in the analysis.

    Parameters
    ----------
    nb_files: int
        Number of files processed by ball batches
    nb_batches: int
        Number of batches in the analysis

    Returns
    -------
    list(int):
    A list representing the number of files processed by each batch in the analysis.

    Examples
    --------
    >>> _compute_batch_sizes(10,4)
    [3,3,2,2]
    """
    """Compute the number of files processed by each batch in the analysis.

    The number of file is equitably distributed among batches.
    Example: 10 files distributed among 4 batches will lead to sizes [3,3,2,2]."""
    base_numbers_of_files = [nb_files // nb_batches] * nb_batches
    remainder_files = nb_files % nb_batches
    return [nb_files+1 if index < remainder_files else nb_files for index,nb_files in enumerate(base_numbers_of_files)]

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
    parser.add_argument("--datetime-begin", "-db", type=str)
    parser.add_argument("--datetime-end", "-de", type=str)
    parser.add_argument("--batch-index", "-i", type=int)
    parser.add_argument("--nb-batches", "-n", type=int)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deletes all existing spectrograms and their zoom levels matching the audio file before processing. If some spectrograms do not match any processed audio files, they will not be deleted.",
    )
    parser.add_argument(
        "--save-matrix",
        "-m",
        action="store_true",
        help="Whether to save the spectrogram matrices or not. Note that activating this parameter might increase greatly the volume of the project.",
    )
    parser.add_argument("--save-for-LTAS", action="store_true")
    parser.add_argument("--spectrogram-metadata-path", type=str, default="")

    args = parser.parse_args()

    print("Parameters :", args)

    os.system("ln -sf /appli/sox/sox-14.4.2_gcc-7.2.0/bin/sox sox")

    dataset = Spectrogram.from_csv(
        dataset_path=args.dataset_path, metadata_csv_path=args.spectrogram_metadata_path
    )

    if not dataset.path.joinpath("processed", "spectrogram", "adjust_metadata.csv"):
        raise FileNotFoundError(
            f"The file adjust_metadata.csv has not been found in the processed/spectrogram folder. Consider using the initialize() or update_parameters() methods."
        )

    datetime_begin = pd.Timestamp(args.datetime_begin)
    datetime_end = pd.Timestamp(args.datetime_end)

    all_processed_files = _get_all_processed_files(datetime_begin=datetime_begin, datetime_end=datetime_end, audio_folder = dataset.audio_path)
    files = _get_files_processed_by_current_batch(all_processed_files, batch_index = args.batch_index, total_number_of_batches = args.nb_batches)

    if missing_files := [file for file in files if not file.exists()]:
        missing_file_list = "\n".join(str(file) for file in missing_files)
        raise FileNotFoundError(
            f"Missing files: {missing_file_list}"
        )

    for audio_file in files:
        print(audio_file)

        dataset.process_file(
            audio_file,
            adjust=False,
            save_matrix=args.save_matrix,
            save_for_LTAS=args.save_for_LTAS,
            clean_adjust_folder=True,
            overwrite=args.overwrite,
        )

    if False and args.save_for_LTAS and args.save_matrix:

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

        print(f"number of welch: {len(files)}")

        for file_npz in files:
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
