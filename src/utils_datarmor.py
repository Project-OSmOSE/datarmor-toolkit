#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from OSmOSE import Spectrogram
from OSmOSE.config import SUPPORTED_AUDIO_FORMAT
from OSmOSE.cluster import reshape
import argparse
import random
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import glob
from typing import List, Union
import shutil
import subprocess
from OSmOSE.utils.audio_utils import get_audio_file
from OSmOSE.utils.core_utils import add_entry_for_APLOSE, select_audio_file


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
        dataset._Dataset__original_folder
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
        print(
            "WARNING: the spectrogram normalization has been changed to spectrum because the data will be normalized using zscore."
        )

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

        reshaped = reshape(
            input_files=files_to_process,
            chunk_size=dataset.spectro_duration,
            new_sr=dataset.dataset_sr,
            output_dir_path=temp_adjustment_output_dir,
            last_file_behavior="discard",
        )

        file_adjust = random.sample(
            get_audio_file(file_path=temp_adjustment_output_dir),
            number_adjustment_spectrogram,
        )

    for audio_file in file_adjust:
        dataset.process_file(audio_file, adjust=True)

    dataset.save_spectro_metadata(spectro_metadata)


def generate_spectro(
    dataset: Spectrogram,
    path_osmose_dataset: Union[str, Path],
    write_datasets_csv_for_APLOSE: bool = False,
    overwrite: bool = False,
    save_matrix: bool = False,
    save_welch: bool = False,
    datetime_begin: str = None,
    datetime_end: str = None,
):

    assert isinstance(
        dataset, Spectrogram
    ), "Not a Spectrogram object passed, adjustment aborted"
    assert isinstance(
        write_datasets_csv_for_APLOSE, bool
    ), "'write_datasets_csv_for_APLOSE' must be a boolean value"
    assert isinstance(overwrite, bool), "'overwrite' must be a boolean value"
    assert isinstance(save_matrix, bool), "'save_matrix' must be a boolean value"
    assert isinstance(save_welch, bool), "'save_welch' must be a boolean value"
    assert isinstance(
        path_osmose_dataset, Union[str, Path]
    ), f"'path_osmose_dataset' must be a path, {path_osmose_dataset} not a valid value"
    assert isinstance(
        datetime_begin, str
    ), f"'datetime_begin' must be a datetime formatted string, {datetime_begin} not a valid value"
    assert isinstance(
        datetime_end, str
    ), f"'datetime_end' must be a datetime formatted string, {datetime_end} not a valid value"

    if write_datasets_csv_for_APLOSE is True:

        file_type = list(
            set([f.suffix for f in get_audio_file(dataset._Dataset__original_folder)])
        )[-1]

        dataset_info = {
            "campaign": dataset.campaign,
            "dataset": dataset.name,
            "spectro_duration": f"{dataset.spectro_duration}",
            "dataset_sr": f"{dataset.dataset_sr}",
            "file_type": file_type,
            "identifier": dataset.campaign
            + "_"
            + dataset.name
            + "_"
            + str(dataset.spectro_duration)
            + "_"
            + str(dataset.dataset_sr),
        }

        dataset_info = pd.DataFrame(dataset_info, index=[0])

        add_entry_for_APLOSE(
            path=path_osmose_dataset, file="datasets_copy.csv", info=dataset_info
        )

    # compute expected_nber_segmented_files
    new_file = list(
        pd.date_range(
            start=pd.Timestamp(datetime_begin),
            end=pd.Timestamp(datetime_end),
            freq=f"{dataset.spectro_duration}s",
        )
    )
    nber_files_to_process = len(new_file) - 1

    batch_size = nber_files_to_process // dataset.batch_number

    dataset.save_spectro_metadata(True)

    for batch in range(dataset.batch_number):
        i_min = batch * batch_size
        i_max = (
            i_min + batch_size
            if batch < dataset.batch_number - 1
            else nber_files_to_process
        )  # If it is the last batch, take all files

        jobfile = dataset.jb.build_job_file(
            script_path=Path(os.path.abspath("../src")).joinpath(
                "qsub_spectrogram_generator_pkg.py"
            ),
            script_args=f"--dataset-path {dataset.path} "
            f"--dataset-sr {dataset.dataset_sr} "
            f"--batch-ind-min {i_min} "
            f"--batch-ind-max {i_max} "
            f"{'--overwrite ' if overwrite else ''}"
            f"{'--save-for-LTAS ' if save_welch else ''}"
            f"{'--save-matrix ' if save_matrix else ''}",
            jobname="OSmOSE_SpectroGenerator",
            preset="low",
            env_name=sys.executable.replace("/bin/python", ""),
            mem="70G",
            walltime="10:00:00",
            logdir=dataset.path.joinpath("log"),
        )

    pending_jobs = [
        jobid
        for jobid in dataset.pending_jobs
        if b"finished"
        not in subprocess.run(["qstat", jobid], capture_output=True).stderr
    ]

    job_id_list = dataset.jb.submit_job(
        dependency=pending_jobs
    )  # submit all built job files
    nb_jobs = len(dataset.jb.finished_jobs) + len(job_id_list)

    if pending_jobs:
        print(f"pending job ids: {pending_jobs}")
    print(f"The job ids are {job_id_list}")


def display_progress(dataset: Spectrogram):

    assert isinstance(
        dataset, Spectrogram
    ), "Not a Spectrogram object passed, display aborted"

    nber_audio_file = len(get_audio_file(dataset.audio_path))
    nber_file_to_process = pd.read_csv(
        str(dataset.audio_path) + "/metadata.csv", header=0
    )["audio_file_count"][0]
    nber_spectro = len(list(dataset.path_output_spectrogram.glob("*png")))
    nber_spectro_to_process = nber_file_to_process * sum(
        2**i for i in range(dataset.zoom_level + 1)
    )

    if nber_audio_file == nber_file_to_process:
        status = "DONE"
    else:
        status = "ONGOING"

    print(
        "o Audio file preparation : " + status + " (",
        nber_audio_file,
        "/",
        str(nber_file_to_process),
        ")",
    )

    if nber_spectro == nber_spectro_to_process:
        status = "DONE"
    else:
        status = "ONGOING"

    print(
        "o Spectrogram generation : " + status + " (",
        nber_spectro,
        "/",
        str(nber_spectro_to_process),
        ")",
    )


def monitor_job(job_id: Union[str, List[str]]):

    assert isinstance(job_id, str) or (
        isinstance(job_id, list) and all(isinstance(item, str) for item in job_id)
    ), f"Job ID must be a string or a list of strings, {job_id} is not a valid value"

    if isinstance(job_id, str):
        job_id = [job_id]

    for j in job_id:
        assert isinstance(j, str), f"Job ID must be a string, {j} is not a valid value"

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
                raise Exception(result.stderr.strip())

            # Extract the job state from the command output
            output_lines = result.stdout.strip().split("\n")
            for line in output_lines:
                if "job_state = " in line:
                    job_state = line.split("=")[1].strip()
                    print(f"o Job ID: {j}\n  Job State: {job_state}")
                    break
            else:
                print(f"o Job ID: {j}\n  Job state not found.")
        except Exception as e:
            print(f"o Job ID: {j}\n  {str(e)}")
