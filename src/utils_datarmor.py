from random import randint
import os
import sys
import pandas as pd
from pathlib import Path
import shutil
import subprocess

from OSmOSE import Spectrogram
from OSmOSE.config import OSMOSE_PATH, print_logger, global_logging_context as glc
from OSmOSE.cluster import reshape
from OSmOSE.utils.audio_utils import get_all_audio_files
from OSmOSE.utils.core_utils import add_entry_for_APLOSE
from OSmOSE.utils.timestamp_utils import strftime_osmose_format


def adjust_spectro(
    dataset: Spectrogram, number_adjustment_spectrogram: int = 1, file_list: [str] = []
):
    """
    Computes adjustment spectrograms.

    Parameters
    ----------
    dataset: Spectrogram
        Spectrogram object
    number_adjustment_spectrogram: int
        Number of adjustment spectrogram to be computed
    """
    assert isinstance(dataset, Spectrogram), "Not a Spectrogram object passed"
    assert (
        isinstance(number_adjustment_spectrogram, int)
        and number_adjustment_spectrogram >= 0
    ), "'number_adjustment_spectrogram' must be an integer >= 0"

    if number_adjustment_spectrogram == 0:
        return


    temp_adjustment_output_dir = (
        dataset.original_folder.parent
        / f"temp_{dataset.spectro_duration}_{dataset.dataset_sr}"
    )

    if (
        dataset.data_normalization == "zscore"
        and dataset.spectro_normalization != "spectrum"
    ):
        dataset.spectro_normalization = "spectrum"
        glc.logger.warning("The spectrogram normalization has been changed to spectrum because the data will be normalized using zscore.")


    file_metadata = pd.read_csv(dataset.original_folder / "file_metadata.csv")

    if len(file_list) > 0:
        if all([f in file_metadata["filename"].values for f in file_list]):
            file_metadata = file_metadata[file_metadata["filename"].isin(file_list)]
        else:
            files_not_in_file_metadata = [
                f for f in file_list if f not in file_metadata["filename"].values
            ]
            raise ValueError(
                f"{files_not_in_file_metadata} not found in file_metadata file."
            )

    if os.path.exists(temp_adjustment_output_dir):
        shutil.rmtree(temp_adjustment_output_dir)

    for _ in range(number_adjustment_spectrogram):
        random_file = file_metadata.sample()  # Gets a random row in file_metadata

        file_start = pd.Timestamp(random_file["timestamp"].values[0])
        file_end = file_start + pd.Timedelta(seconds=random_file["duration"].values[0])

        spectro_start = file_start + pd.Timedelta(
            seconds=randint(
                0, int((file_end - file_start).seconds - dataset.spectro_duration)
            )
        )
        spectro_stop = spectro_start + pd.Timedelta(dataset.spectro_duration, "s")

        reshape(
            input_files=dataset.original_folder,
            segment_size=dataset.spectro_duration,
            new_sr=dataset.dataset_sr,
            output_dir_path=temp_adjustment_output_dir,
            datetime_begin=str(spectro_start),
            datetime_end=str(spectro_stop),
        )

    files_adjust = get_all_audio_files(temp_adjustment_output_dir)

    for audio_file in files_adjust:
        dataset.process_file(audio_file, adjust=True)


def generate_spectro(
    dataset: Spectrogram,
    path_osmose_dataset: str | Path,
    write_datasets_csv_for_aplose: bool = False,
    overwrite: bool = False,
    save_matrix: bool = False,
    save_welch: bool = False,
    datetime_begin: str = None,
    datetime_end: str = None,
):
    """
    Computes spectrograms of a given dataset.

    Parameters
    ----------
    dataset: Spectrogram
        Spectrogram object
    path_osmose_dataset: str | Path
        Path to dataset
    write_datasets_csv_for_aplose: bool
        Boolean to add or not the dataset to APLOSE dataset csv file
    overwrite: bool
        Force overwrite of existing dataset files
    save_matrix: bool
        Save matrix of spectrograms
    save_welch: bool
        Save welch spectrograms
    datetime_begin: str
        Begin datetime of spectrograms to compute
    datetime_end: str
        End datetime of spectrograms to compute
    """
    first_job = next(iter(dataset.jb.all_jobs), None)
    log_dir = (
        (dataset.path / OSMOSE_PATH.log)
        if first_job is None
        else first_job["path"].parent
    )

    assert isinstance(
        dataset, Spectrogram
    ), "Not a Spectrogram object passed, adjustment aborted"
    assert isinstance(
        write_datasets_csv_for_aplose, bool
    ), "'write_datasets_csv_for_APLOSE' must be a boolean value"
    assert isinstance(overwrite, bool), "'overwrite' must be a boolean value"
    assert isinstance(save_matrix, bool), "'save_matrix' must be a boolean value"
    assert isinstance(save_welch, bool), "'save_welch' must be a boolean value"
    assert isinstance(dataset.concat, bool), "'concat' must be a boolean value"
    assert isinstance(
        path_osmose_dataset, str | Path
    ), f"'path_osmose_dataset' must be a path, {path_osmose_dataset} not a valid value"

    datetime_begin, datetime_end = _clip_timestamps(begin=datetime_begin, end=datetime_end, dataset=dataset)

    datetime_begin = strftime_osmose_format(datetime_begin)
    datetime_end = strftime_osmose_format(datetime_end)

    if write_datasets_csv_for_aplose is True:

        file_type = list(
            set([f.suffix for f in get_all_audio_files(dataset.original_folder)])
        )[-1]

        dataset_info = {
            "path": dataset.path.relative_to(path_osmose_dataset),
            "dataset": dataset.name,
            "spectro_duration": f"{dataset.spectro_duration}",
            "dataset_sr": f"{dataset.dataset_sr}",
            "file_type": file_type,
        }

        dataset_info = pd.DataFrame(dataset_info, index=[0])

        add_entry_for_APLOSE(
            path=path_osmose_dataset, file="datasets.csv", info=dataset_info
        )
    temp_adjustment_output_dir = (
            dataset.audio_path.parent
            / f"temp_{dataset.spectro_duration}_{dataset.dataset_sr}"
    )

    if os.path.exists(temp_adjustment_output_dir):
        shutil.rmtree(temp_adjustment_output_dir)

    job_files = []

    dataset.prepare_paths()
    
    spectrogram_metadata_path = dataset.save_spectro_metadata(False)

    for batch in range(dataset.batch_number):

        job_file = dataset.jb.build_job_file(
            script_path=Path(os.path.abspath("../src")).joinpath(
                "qsub_spectrogram_generator_pkg.py"
            ),
            script_args=f"--dataset-path {dataset.path} "
            f"--dataset-sr {dataset.dataset_sr} "
            f"--spectrogram-metadata-path {spectrogram_metadata_path} "
            f"--datetime-begin {datetime_begin} "
            f"--datetime-end {datetime_end} "
            f"--batch-index {batch} "
            f"--nb-batches {dataset.batch_number} "
            f"{'--overwrite ' if overwrite else ''}"
            f"{'--save-for-LTAS ' if save_welch else ''}"
            f"{'--save-matrix ' if save_matrix else ''}",
            jobname="OSmOSE_SpectroGenerator",
            preset="low",
            env_name=sys.executable.replace("/bin/python", ""),
            mem="70G",
            walltime="10:00:00",
            logdir=log_dir,
        )

    if hasattr(dataset, "pending_jobs"):
        pending_jobs = [
            jobid
            for jobid in dataset.pending_jobs
            if b"finished"
            not in subprocess.run(["qstat", jobid], capture_output=True).stderr
        ]
    else:
        pending_jobs = []

    job_id_list = dataset.jb.submit_job(dependency=pending_jobs) # submit all built job files
    nb_jobs = len(dataset.jb.finished_jobs) + len(job_id_list)

    if pending_jobs:
        glc.logger.info(f"pending job ids: {pending_jobs}")
    glc.logger.info(f"The job ids are {job_id_list}")

@glc.set_logger(print_logger)
def display_progress(
    dataset: Spectrogram,
    datetime_begin: pd.Timestamp | None = None,
    datetime_end: pd.Timestamp | None = None,
):
    """
    Display progress on audio segments and spectrogram computation.

    Parameters
    ----------
    dataset: Spectrogram
        Spectrogram object
    datetime_begin: pd.Timestamp
        Begin datetime of spectrograms to compute
    datetime_end: pd.Timestamp
        End datetime of spectrograms to compute
    """
    assert isinstance(
        dataset, Spectrogram
    ), "Not a Spectrogram object passed, display aborted"

    datetime_begin, datetime_end = _clip_timestamps(begin=datetime_begin, end=datetime_end, dataset=dataset)

    assert isinstance(
        datetime_begin, pd.Timestamp
    ), f"'{datetime_begin}' not a valid timestamp"
    assert isinstance(
        datetime_end, pd.Timestamp
    ), f"'{datetime_end}' not a valid timestamp"

    dataset.prepare_paths()
    target_nb_files = _approximate_target_nb_file(datetime_begin, datetime_end, dataset)

    number_spectro = len(list(dataset.path_output_spectrogram.glob("*png")))
    number_audio_file = len(list(dataset.audio_path.glob("*.wav")))
    number_spectro_to_process = target_nb_files * sum(
        2**i for i in range(dataset.zoom_level + 1)
    )

    if number_audio_file == target_nb_files:
        status = "DONE"
        dataset.jb.update_job_status()
        dataset.jb.update_job_access()
    else:
        status = "ONGOING"

    glc.logger.info(f"o Audio file preparation : {status} ({number_audio_file}/{target_nb_files})")

    glc.logger.info(f"\t- Generated audio: {len(get_all_audio_files(dataset.audio_path))}")

    if number_spectro == number_spectro_to_process:
        status = "DONE"
        dataset.jb.update_job_status()
        dataset.jb.update_job_access()
    else:
        status = "ONGOING"

    glc.logger.info(f"o Spectrogram generation : {status} ({number_spectro}/{number_spectro_to_process})")

@glc.set_logger(print_logger)
def monitor_job(dataset: Spectrogram):
    """
    Monitor ongoing jobs

    Parameters
    ----------
    dataset: Spectrogram
        Spectrogram object
    """
    assert isinstance(
        dataset, Spectrogram
    ), "Not a Spectrogram object passed, display aborted"

    prepared_jobs = [job["id"] for job in dataset.jb.prepared_jobs]
    ongoing_jobs = [job["id"] for job in dataset.jb.ongoing_jobs]
    cancelled_jobs = [job["id"] for job in dataset.jb.cancelled_jobs]
    finished_jobs = [job["id"] for job in dataset.jb.finished_jobs]
    all_jobs = prepared_jobs + ongoing_jobs + cancelled_jobs + finished_jobs

    for j in all_jobs:
        try:
            # Run the qstat command
            result = subprocess.run(
                ["qstat", "-f", j],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Check for errors
            if result.returncode != 0:
                glc.logger.error("The qstat command threw an error.")
                raise Exception(result.stderr.strip())

            # Extract the job state from the command output
            output_lines = result.stdout.strip().split("\n")
            for line in output_lines:
                if "job_state = " in line:
                    job_state = line.split("=")[1].strip()
                    glc.logger.info(f"Job ID: {j}\n  Job State: {job_state}")
                    break
            else:
                glc.logger.info(f"o Job ID: {j}\n  Job state not found.")
        except Exception as e:
            glc.logger.error(f"o Job ID: {j}\n  {str(e)}")
            raise e


@glc.set_logger(print_logger)
def read_job(job_id: str, dataset: Spectrogram):
    """
    Inspect job status

    Parameters
    ----------
    job_id: str
        ID of the job to display
    dataset: Spectrogram
        Spectrogram object
    """

    assert isinstance(
        dataset, Spectrogram
    ), "Not a Spectrogram object passed, display aborted"
    assert isinstance(job_id, str), "'job_id' must be a string"

    finished_jobs = dataset.jb.finished_jobs
    id_jobs = [job["id"] for job in finished_jobs]
    outfile = next(
        (job["outfile"] for job in finished_jobs if job["id"] == job_id), None
    )

    if outfile:
        if outfile.exists():
            with open(outfile) as f:
                glc.logger.info(f.read())
        else:
            raise FileNotFoundError
    else:
        glc.logger.info(f"{job_id} not in finished jobs")

def _clip_timestamps(begin: pd.Timestamp or str or None, end: pd.Timestamp or str or None, dataset: Spectrogram):

    file_metadata = pd.read_csv(
        dataset.original_folder / "file_metadata.csv", parse_dates=["timestamp"]
    )

    if not begin:
        begin = file_metadata["timestamp"].iloc[0]
    else:
        try:
            begin = pd.Timestamp(begin)
        except Exception as e:
            raise ValueError(f"'datetime_begin' not a valid datetime: {e}")

    if not end:
        end = file_metadata["timestamp"].iloc[-1] + pd.Timedelta(
            file_metadata["duration"].iloc[-1], unit="s"
        )
    else:
        try:
            end = pd.Timestamp(end)
        except Exception as e:
            raise ValueError(f"'datetime_end' not a valid datetime: {e}")
    return begin, end

def _approximate_target_nb_file(begin: pd.Timestamp, end: pd.Timestamp, dataset: Spectrogram) -> int:
    file_duration = dataset.spectro_duration
    target_file_time = [(start, start + pd.Timedelta(seconds = file_duration)) for start in pd.date_range(begin, end, freq=pd.Timedelta(seconds=file_duration))]
    file_metadata = pd.read_csv(dataset.original_folder / "file_metadata.csv")
    file_metadata["duration"] = file_metadata["duration"].apply(lambda d: pd.Timedelta(seconds = d))
    file_metadata["timestamp"] = file_metadata["timestamp"].apply(lambda t: pd.Timestamp(t))
    file_metadata["end"] = file_metadata["timestamp"] + file_metadata["duration"]

    target_file_time = [(begin,end) for begin,end in target_file_time if len(file_metadata.loc[(file_metadata["timestamp"] < end ) & (file_metadata["end"] > begin )]) > 0]
    return len(target_file_time)