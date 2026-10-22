#!/bin/bash

# phase 1
kdfp run -c kdfp/configs/base_100m_tokens.yaml

# phase 2
kdfp run -c kdfp/configs/off_proj_search.yaml

# phase 3
kdfp run -c kdfp/configs/jsd_beta_search.yaml

# phase 4
kdfp run -c kdfp/configs/objective_search_4A.yaml
kdfp run -c kdfp/configs/objective_search_4B_additive.yaml
kdfp run -c kdfp/configs/objective_search_4B_subtractive.yaml

# phase 5
kdfp run -c kdfp/configs/final_test_1.yaml
kdfp run -c kdfp/configs/final_test_2.yaml
kdfp run -c kdfp/configs/final_test_3_down.yaml
