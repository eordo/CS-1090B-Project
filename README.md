# Neural CDE GDP Nowcasting

**CS 1090B Advanced Topics in Data Science**

_Group #2: Grant Solomon & Eric Ordonez_

## Setup

### FRED

Querying FRED data requires an API key, which can be requested [here](https://fred.stlouisfed.org/docs/api/fred/v2/api_key.html) for free.
Create an `.env` file in the root directory and save your API key in it like so:

```bash
# .env
export FRED_API_KEY="my_api_key"
```

### NCDENow

This project uses the NCDENow nowcasting framework [(Lim et al., 2024)](https://arxiv.org/abs/2409.08732).
The code can be found in [this GitHub repository](https://github.com/sklim84/NCDENow_CIKM2024), and it is included here as a submodule.

```bash
# If you have not already cloned this repo
git clone --recurse-submodules https://github.com/eordo/CS-1090B-Project.git

# If you already cloned but without --recurse-submodules
git submodule update --init --recursive
```
