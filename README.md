<div align="center">

  <img src="assets/osmose_logo.png" height="80px">
  <img src="assets/ifremer_logo.jpg" height="80px">
</div>

# OSmOSE Project on Datarmor

The [Open Science meets Ocean Sound Explorers](https://github.com/Project-OSmOSE) is a collaborative research project aiming to develop a complete collection of FAIR acoustic analysis tools and methods. 

## Presentation

A [toolkit](https://github.com/Project-OSmOSE/OSEkit) has been deployed on the [DATARMOR](https://www.ifremer.fr/fr/infrastructures-de-recherche/le-supercalculateur-datarmor) cluster of IFREMER, on which our production version runs. The toolkit is available to Datarmor members as a suite of notebooks covering basic processing cases:

1. `data_uploader.ipynb` : used for the importation and formatting of a new dataset;

2. `spectrogram_generator.ipynb` : used for the generation of file-scale (or shorter) spectrograms;

3. `soundscape_analyzer.ipynb` : used for long-term analysis (i.e. with timescale at least longer than the audio file duration), including the computation of soundscape metrics (e.g. long-term averaged spectrograms, EPD) and the retrieval of raw welch spectra at different time resolutions;

Note : Steps 1 and 2 can be done for a batch of datasets at once using `wrapper_data_uploader.ipynb` and `wrapper_spectrogram_generation.ipynb`

See our [user guide](https://project-osmose.github.io/OSEkit/) for more details.

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. See [LICENSE](https://www.gnu.org/licenses/licenses.html) for the complete AGPL license






