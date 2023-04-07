import argparse
from OSmOSE import Spectrogram
import random
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python script to process a few files and adjust spectrogram parameters.")
    required = parser.add_argument_group('required arguments')
    parser.add_argument("--input-file-list", "-l", help="The list of audio file names separated by a space. If not provided, will pick `--nb-files` files at random.")
    parser.add_argument("--nb-files", "-n", type=int, help="Number of files to pick at random in the dataset. If not provided, will pick one file at random.")
    required.add_argument("--sr-analysis", "-s", required=True, help="The analysis frequency.")
    required.add_argument("--dataset-path", "-p", required=True, help="The path to the dataset folder")

    args = parser.parse_args()
    os.system("ln -sf /appli/sox/sox-14.4.2_gcc-7.2.0/bin/sox sox")
    dataset = Spectrogram(args.dataset_path, sr_analysis=args.sr_analysis)

    all_files = [wav_file for wav_file in dataset.path.joinpath("data","audio", f"{str(dataset.spectro_duration)}_{str(args.sr_analysis)}").glob("*wav")]

    if args.input_file_list:
        file_list = args.input_file_list.split(" ")
    elif args.nb_files:
        file_list = random.sample(all_files, args.nb_files)
    else:
        file_list = random.choice(all_files)

    for file in file_list:
        dataset.process_file(file, adjust=True)