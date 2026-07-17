### Test Only ###
import os
from pathlib import Path

from eoh.src.eoh import eoh
from eoh.src.eoh.utils.getParas import Paras

SCRIPT_DIR = Path(__file__).resolve().parent

# Parameter initialization #
paras = Paras() 

# Set parameters #
paras.set_paras(method = "lpbs",
                problem = "mnv",
                llm_api_endpoint = "api.deepseek.com", # set endpoint
                llm_api_key = os.environ.get("DEEPSEEK_API_KEY", ""),   # set your key
                llm_model = "deepseek-v4-flash", # set llm
                ec_pop_size = 10,
                ec_n_pop = 20,
                exp_n_proc = 1,
                exp_debug_mode = False,
                exp_use_seed = True,
                exp_seed_path = str(SCRIPT_DIR.parent / "problems" / "mnv" / "seeds.json"),
                exp_output_path = str(SCRIPT_DIR))

# EoH initilization
evolution = eoh.EVOL(paras)

# run EoH
evolution.run_multiple(5)


