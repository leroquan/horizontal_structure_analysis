# Data locations

MITgcm dataset paths and loading parameters are defined in [config.json](./config.json), located in the same directory as this file.

The configuration is organized by machine hostname, then dataset name. Select the appropriate machine and dataset entry:

- `datapath`: directory containing the model result files.
- `gridpath`: directory containing the grid files.
- `ref_date`, `dt`, and `endian`: time reference, model timestep, and binary byte order used when loading results.

Use the configuration values directly rather than duplicating paths here.
