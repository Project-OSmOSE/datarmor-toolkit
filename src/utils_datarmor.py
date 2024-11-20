#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from OSmOSE import Spectrogram
from OSmOSE.config import SUPPORTED_AUDIO_FORMAT, OSMOSE_PATH, global_logging_context as glc
from OSmOSE.cluster import reshape
import random
import os
import re
import sys
import pandas as pd
from pathlib import Path
from typing import List, Union
import shutil
import subprocess
from OSmOSE.utils.audio_utils import get_all_audio_files
from OSmOSE.utils.core_utils import add_entry_for_APLOSE


def validate_datetime(
    dataset: Spectrogram,
    datetime_begin: Union[str, None] = None,
    datetime_end: Union[str, None] = None,
):

    # Validate datetimes format
    regex = r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[+-]\d{4}$"
    regex2 = r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$"
    datetime_format_error = (
        "Please use the following format: 'YYYY-MM-DDTHH:MM:SS+/-HHMM'."
    )

    file_metadata_path = dataset.original_folder / "file_metadata.csv"
    file_metadata = pd.read_csv(file_metadata_path, parse_dates=["timestamp"])

    if datetime_begin:
        if not re.match(regex, datetime_begin) and not re.match(regex2, datetime_begin):
            raise ValueError(
                f"Invalid format for datetime_begin. {datetime_format_error}"
            )
        datetime_begin = pd.Timestamp(datetime_begin)
    else:
        datetime_begin = file_metadata["timestamp"].iloc[0]

    if datetime_end:
        if not re.match(regex, datetime_end) and not re.match(regex2, datetime_end):
            raise ValueError(
                f"Invalid format for datetime_end. {datetime_format_error}"
            )
        datetime_end = pd.Timestamp(datetime_end)
    else:
        datetime_end = file_metadata["timestamp"].iloc[-1] + pd.Timedelta(
            file_metadata["duration"].iloc[-1], unit="s"
        )

    return datetime_begin, datetime_end


def adjust_spectro(
    dataset: Spectrogram,
    file_list: List[str] = [],
    number_adjustment_spectrogram: int = 1,
    spectro_metadata: bool = True,
):

    assert isinstance(
        dataset, Spectrogram
    ), "Not a Spectrogram object passed, adjustment aborted"
    assert (
        isinstance(number_adjustment_spectrogram, int)
        and number_adjustment_spectrogram >= 0
    ), "'number_adjustment_spectrogram' must be an integer >= 0"
    assert isinstance(
        spectro_metadata, bool
    ), "'spectro_metadata' must be a boolean value"

    if number_adjustment_spectrogram == 0:
        return

    dataset.audio_path = (
        dataset.original_folder
    )  # necessary because of the reshape thereafter that changes the audio_path

    orig_metadata = pd.read_csv(
        dataset._get_original_after_build().joinpath("metadata.csv"), header=0
    )
    orig_dura = orig_metadata["audio_file_origin_duration"][0]
    orig_sr = orig_metadata["origin_sr"][0]

    origin_files = []
    for ext in SUPPORTED_AUDIO_FORMAT:
        origin_files_ext = list(dataset.audio_path.glob(f"*{ext}"))
        [origin_files.append(f) for f in origin_files_ext]

    temp_adjustment_output_dir = (
        dataset.audio_path.parent
        / f"temp_{dataset.spectro_duration}_{dataset.dataset_sr}"
    )

    if (
        dataset.data_normalization == "zscore"
        and dataset.spectro_normalization != "spectrum"
    ):
        dataset.spectro_normalization = "spectrum"
        glc.logger.warning("The spectrogram normalization has been changed to spectrum because the data will be normalized using zscore.")


    if len(file_list) > 0:
        files_adjust = [temp_adjustment_output_dir + "/" + ff for ff in file_list]

    elif dataset.spectro_duration == orig_dura and dataset.dataset_sr == orig_sr:
        files_adjust = random.sample(
            origin_files, min(number_adjustment_spectrogram, len(origin_files))
        )
        files_adjust = files_adjust[:number_adjustment_spectrogram]

    else:
        files_to_process = random.sample(
            origin_files, min(number_adjustment_spectrogram, len(origin_files))
        )

        if os.path.exists(temp_adjustment_output_dir):
            shutil.rmtree(temp_adjustment_output_dir)

        reshape(
            input_files=files_to_process,
            segment_size=dataset.spectro_duration,
            new_sr=dataset.dataset_sr,
            output_dir_path=temp_adjustment_output_dir,
            concat=dataset.concat,
        )

        files_adjust = random.sample(
            get_all_audio_files(directory = temp_adjustment_output_dir),
            number_adjustment_spectrogram,
        )

    for audio_file in files_adjust:
        dataset.process_file(audio_file, adjust=True)

    dataset.save_spectro_metadata(spectro_metadata)


def generate_spectro(
    dataset: Spectrogram,
    path_osmose_dataset: Union[str, Path],
    write_datasets_csv_for_aplose: bool = False,
    overwrite: bool = False,
    save_matrix: bool = False,
    save_welch: bool = False,
    datetime_begin: Union[str, None] = None,
    datetime_end: Union[str, None] = None,
):

    datetime_begin, datetime_end = validate_datetime(
        dataset, datetime_begin, datetime_end
    )

    first_job = next(iter(dataset.jb.all_jobs), None)
    log_dir = (dataset.path / OSMOSE_PATH.log) if first_job is None else first_job["path"].parent

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
        path_osmose_dataset, Union[str, Path]
    ), f"'path_osmose_dataset' must be a path, {path_osmose_dataset} not a valid value"
    assert isinstance(
        datetime_begin, pd.Timestamp
    ), f"'datetime_begin' must be either 'None' or a datetime, {datetime_begin} not a valid value"
    assert isinstance(
        datetime_end, pd.Timestamp
    ), f"'datetime_end' must be either 'None' or a datetime, {datetime_end} not a valid value"

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

    jobfiles = []

    dataset.prepare_paths()

    files = _files_in_analysis(datetime_begin=datetime_begin, datetime_end=datetime_end, audio_folder=dataset.audio_path)
    batch_sizes = _compute_batch_sizes(nb_files = len(files), nb_batches = dataset.batch_number)
    batch_indexes = [sum(batch_sizes[:i]) for i in range(len(batch_sizes))]
    
    spectrogram_metadata_path = dataset.save_spectro_metadata(False)

    for batch in range(len(batch_indexes)):
        first_file_index = batch_indexes[batch]
        last_file_index = first_file_index + batch_sizes[batch]

        jobfile = dataset.jb.build_job_file(
            script_path=Path(os.path.abspath("../src")).joinpath(
                "qsub_spectrogram_generator_pkg.py"
            ),
            script_args=f"--dataset-path {dataset.path} "
            f"--dataset-sr {dataset.dataset_sr} "
            f"--spectrogram-metadata-path {spectrogram_metadata_path} "                        
            f"--files {' '.join(files[first_file_index:last_file_index])} "
            f"--first-file-index {first_file_index} "
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

        jobfiles.append(jobfile)

    if hasattr(dataset, "pending_jobs"):
        pending_jobs = [
            jobid
            for jobid in dataset.pending_jobs
            if b"finished"
            not in subprocess.run(["qstat", jobid], capture_output=True).stderr
        ]
    else:
        pending_jobs = []

    job_id_list = [dataset.jb.submit_job(
        jobfile = jobfile,
        dependency=pending_jobs
    ) for jobfile in jobfiles]  # submit all built job files
    nb_jobs = len(dataset.jb.finished_jobs) + len(job_id_list)

    if pending_jobs:
        glc.logger.info(f"pending job ids: {pending_jobs}")
    glc.logger.info(f"The job ids are {job_id_list}")


def display_progress(dataset: Spectrogram, datetime_begin: str, datetime_end: str):

    datetime_begin, datetime_end = validate_datetime(
        dataset, datetime_begin, datetime_end
    )

    assert isinstance(
        dataset, Spectrogram
    ), "Not a Spectrogram object passed, display aborted"
    assert isinstance(
        datetime_begin, pd.Timestamp
    ), f"'datetime_begin' must be either 'None' or a datetime, {datetime_begin} not a valid value"
    assert isinstance(
        datetime_end, pd.Timestamp
    ), f"'datetime_end' must be either 'None' or a datetime, {datetime_end} not a valid value"

    nber_audio_file = len(get_all_audio_files(dataset.audio_path))

    if dataset.concat:
        test_range = pd.date_range(
            start=datetime_begin,
            end=datetime_end,
            freq=f"{dataset.spectro_duration}s",
        ).to_list()
        origin_dt = pd.read_csv(
            dataset.path_input_audio_file / "timestamp.csv", parse_dates=["timestamp"]
        )["timestamp"]
        nber_file_to_process = 0
        for dt in test_range:
            if dt >= origin_dt.iloc[0] - pd.Timedelta(
                dataset.spectro_duration, "s"
            ) and dt <= origin_dt.iloc[-1] + pd.Timedelta(
                dataset.spectro_duration, "s"
            ):
                nber_file_to_process += 1

        nber_spectro = len(list(dataset.path_output_spectrogram.glob("*png")))
        nber_spectro_to_process = nber_file_to_process * sum(
            2**i for i in range(dataset.zoom_level + 1)
        )
    else:
        # nber_file_to_process = len(dataset.list_audio_to_process)
        origin_timestamp = pd.read_csv(
            dataset.original_folder / "timestamp.csv", parse_dates=["timestamp"]
        )
        datetime_begin = (
            origin_timestamp["timestamp"].iloc[0]
            if datetime_begin is None
            else datetime_begin
        )
        datetime_end = (
            origin_timestamp["timestamp"].iloc[-1]
            if datetime_end is None
            else datetime_end
        )
        nber_file_to_process = len(
            origin_timestamp[
                (origin_timestamp["timestamp"] >= datetime_begin)
                & (origin_timestamp["timestamp"] <= datetime_end)
            ]
        )

    nber_spectro = len(list(dataset.path_output_spectrogram.glob("*png")))
    nber_spectro_to_process = nber_file_to_process * sum(
        2**i for i in range(dataset.zoom_level + 1)
    )

    # counting the skipped files
    out_file = [
        str(job["outfile"])
        for job in dataset.jb.finished_jobs
        if "reshape" in str(job["outfile"])
    ]
    skipped = 0
    if out_file:
        for file in out_file:
            with open(file, "r") as f:
                skipped += sum(line.count("Skipping...") for line in f)
        nber_audio_file += skipped

    if nber_audio_file == nber_file_to_process:
        status = "DONE"
        dataset.jb.update_job_status()
        dataset.jb.update_job_access()
    else:
        status = "ONGOING"

    glc.logger.info(f"o Audio file preparation : {status} ({nber_audio_file}/{nber_file_to_process})")

    glc.logger.info(f"\t- Generated audio: {len(get_all_audio_files(dataset.audio_path))}")
    glc.logger.info(f"\t- Discarded audio: {skipped}")

    if nber_spectro == nber_spectro_to_process:
        status = "DONE"
        dataset.jb.update_job_status()
        dataset.jb.update_job_access()
    else:
        status = "ONGOING"

    glc.logger.info(f"o Spectrogram generation : {status} ({nber_spectro}/{nber_spectro_to_process})")

def _files_in_analysis(datetime_begin: pd.Timestamp, datetime_end: pd.Timestamp, audio_folder: Path) -> list[str]:
    timestamps = pd.read_csv(audio_folder / "timestamp.csv")
    file_duration = float(pd.read_csv(audio_folder / "metadata.csv")["audio_file_dataset_duration"][0])
    timestamps["begin"] = timestamps["timestamp"].apply(lambda t: pd.Timestamp(t))
    timestamps["end"] = timestamps["begin"] + pd.Timedelta(seconds=file_duration)
    return [str(audio_folder/filename) for filename in timestamps.loc[(datetime_begin < timestamps["end"]) & (datetime_end > timestamps["begin"]), "filename"]]

def _compute_batch_sizes(nb_files: int, nb_batches: int):
    base_numbers_of_files = [nb_files // nb_batches] * nb_batches
    remainder_files = nb_files % nb_batches
    return [nb_files+1 if index < remainder_files else nb_files for index,nb_files in enumerate(base_numbers_of_files)]

def monitor_job(dataset: Spectrogram):

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


def read_job(job_id: str, dataset: Spectrogram):

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
