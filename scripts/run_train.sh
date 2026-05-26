#!/bin/bash
#===============================================================================
# MACE Training / Fine-Tuning Script
# All flags from build_default_arg_parser (mace/mace/tools/arg_parser.py)
#
# Usage: Fill in the values between < > brackets, then run:
#   bash run_train.sh
#
# To keep things simple, build args with FLAGS+=() as shown.
# Uncomment individual flags as needed.
#
# Boolean flags: use --flag or --no-flag (e.g. --amsgrad, --no-amsgrad).
#===============================================================================
set -euo pipefail

# -------------------------------------------------------------------
# Name & Seed
# -------------------------------------------------------------------
FLAGS=()
FLAGS+=(--name "<experiment_name>")
FLAGS+=(--seed <123>)

# -------------------------------------------------------------------
# Directories
# -------------------------------------------------------------------
FLAGS+=(--work_dir "<.>")
FLAGS+=(--log_dir "<path/to/logs>")
FLAGS+=(--model_dir "<path/to/models>")
FLAGS+=(--checkpoints_dir "<path/to/checkpoints>")
FLAGS+=(--results_dir "<path/to/results>")
FLAGS+=(--downloads_dir "<path/to/downloads>")

# -------------------------------------------------------------------
# Device & Logging
# -------------------------------------------------------------------
FLAGS+=(--device "<cpu|cuda|mps|xpu>")
FLAGS+=(--default_dtype "<float32|float64>")
# FLAGS+=(--distributed)
FLAGS+=(--launcher "<slurm|torchrun|mpi|none>")
FLAGS+=(--log_level "<INFO>")
FLAGS+=(--plot <true>)
FLAGS+=(--plot_frequency <0>)
FLAGS+=(--plot_interaction_e <false>)
FLAGS+=(--error_table "<PerAtomRMSE>")

# -------------------------------------------------------------------
# Model Architecture
# -------------------------------------------------------------------
FLAGS+=(--model "<MACE>")
FLAGS+=(--r_max <5.0>)
FLAGS+=(--radial_type "<bessel>")
FLAGS+=(--num_radial_basis <8>)
FLAGS+=(--num_cutoff_basis <5>)
# FLAGS+=(--pair_repulsion)
FLAGS+=(--distance_transform "<None>")
FLAGS+=(--apply_cutoff <true>)
FLAGS+=(--use_last_readout_only <false>)
FLAGS+=(--use_embedding_readout <false>)
FLAGS+=(--interaction "<RealAgnosticResidualInteractionBlock>")
FLAGS+=(--interaction_first "<RealAgnosticResidualInteractionBlock>")
FLAGS+=(--max_ell <3>)
FLAGS+=(--correlation <3>)
FLAGS+=(--use_reduced_cg <false>)
FLAGS+=(--use_so3 <false>)
FLAGS+=(--use_agnostic_product <false>)
FLAGS+=(--num_interactions <2>)
FLAGS+=(--MLP_irreps "<16x0e>")
FLAGS+=(--radial_MLP "<[64, 64, 64]>")
# FLAGS+=(--hidden_irreps "<16x0e>")
# FLAGS+=(--edge_irreps "<16x0e>")
FLAGS+=(--use_edge_irreps_first <false>)
FLAGS+=(--num_channels <128>)
FLAGS+=(--max_L <1>)
FLAGS+=(--gate "<silu>")

# -------------------------------------------------------------------
# Scaling
# -------------------------------------------------------------------
FLAGS+=(--scaling "<rms_forces_scaling>")
FLAGS+=(--avg_num_neighbors <1.0>)
FLAGS+=(--compute_avg_num_neighbors <true>)
FLAGS+=(--compute_stress <false>)
FLAGS+=(--compute_forces <true>)
FLAGS+=(--compute_polarizability <false>)
FLAGS+=(--compute_atomic_dipole <false>)

# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
FLAGS+=(--train_file "<path/to/train.xyz>")
FLAGS+=(--valid_file "<path/to/valid.xyz>")
FLAGS+=(--valid_fraction <0.1>)
FLAGS+=(--test_file "<path/to/test.xyz>")
FLAGS+=(--test_dir "<path/to/test_dir>")
FLAGS+=(--multi_processed_test <false>)
FLAGS+=(--num_workers <0>)
FLAGS+=(--pin_memory <true>)
# FLAGS+=(--atomic_numbers "<[6,1,8]>")
# FLAGS+=(--mean <0.0>)
# FLAGS+=(--std <0.0>)
# FLAGS+=(--statistics_file "<path/to/stats.json>")
# FLAGS+=(--les_arguments "<path/to/les_config.yaml>")
# FLAGS+=(--E0s '{"H": -0.5, "O": -5.0}')

# -------------------------------------------------------------------
# Fine-tuning (uncomment as needed)
# -------------------------------------------------------------------
# FLAGS+=(--foundation_model "<path/to/foundation_model.pt>")
# FLAGS+=(--foundation_model_kwargs '{}')
# FLAGS+=(--no-foundation_model_readout)
# FLAGS+=(--foundation_head "<head_name>")
# FLAGS+=(--foundation_filter_elements <true>)
# FLAGS+=(--foundation_model_elements <false>)
# FLAGS+=(--heads '{"head1": {"train_file": "path1.xyz", "E0s": {"H": -0.5}}}')
# FLAGS+=(--multiheads_finetuning <true>)
# FLAGS+=(--weight_pt_head <1.0>)
# FLAGS+=(--real_pt_data_ratio_threshold <0.1>)
# FLAGS+=(--num_samples_pt <10000>)
# FLAGS+=(--pt_train_file "<path/to/pt_train.xyz>")
# FLAGS+=(--pt_valid_file "<path/to/pt_valid.xyz>")
# FLAGS+=(--subselect_pt "<random>")
# FLAGS+=(--filter_type_pt "<none>")
# FLAGS+=(--no-allow_random_padding_pt)
# FLAGS+=(--force_mh_ft_lr <false>)
# FLAGS+=(--pseudolabel_replay <false>)
# FLAGS+=(--pseudolabel_replay_compute_stress <false>)
# FLAGS+=(--keep_isolated_atoms <false>)
# FLAGS+=(--lora <false>)
# FLAGS+=(--lora_rank <4>)
# FLAGS+=(--lora_alpha <1.0>)

# -------------------------------------------------------------------
# Data Keys
# -------------------------------------------------------------------
FLAGS+=(--energy_key "<energy>")
FLAGS+=(--forces_key "<forces>")
FLAGS+=(--virials_key "<virials>")
FLAGS+=(--stress_key "<stress>")
FLAGS+=(--dipole_key "<dipole>")
FLAGS+=(--polarizability_key "<polarizability>")
FLAGS+=(--head_key "<head>")
FLAGS+=(--charges_key "<charges>")
FLAGS+=(--elec_temp_key "<elec_temp>")
FLAGS+=(--total_spin_key "<total_spin>")
FLAGS+=(--total_charge_key "<total_charge>")
# FLAGS+=(--embedding_specs '{"total_spin": {"type": "categorical", "per": "graph", "num_classes": 101, "emb_dim": 64}}')
FLAGS+=(--skip_evaluate_heads "<pt_head>")

# -------------------------------------------------------------------
# Loss & Optimization
# -------------------------------------------------------------------
FLAGS+=(--loss "<weighted>")
FLAGS+=(--forces_weight <100.0>)
FLAGS+=(--swa_forces_weight <100.0>)
FLAGS+=(--energy_weight <1.0>)
FLAGS+=(--swa_energy_weight <1000.0>)
FLAGS+=(--virials_weight <1.0>)
FLAGS+=(--swa_virials_weight <10.0>)
FLAGS+=(--stress_weight <1.0>)
FLAGS+=(--swa_stress_weight <10.0>)
FLAGS+=(--dipole_weight <1.0>)
FLAGS+=(--swa_dipole_weight <1.0>)
FLAGS+=(--polarizability_weight <1.0>)
FLAGS+=(--swa_polarizability_weight <1.0>)
FLAGS+=(--config_type_weights '{"Default": 1.0}')
FLAGS+=(--huber_delta <0.01>)
FLAGS+=(--optimizer "<adam>")
FLAGS+=(--beta <0.9>)
FLAGS+=(--beta1_schedulefree <0.9>)
FLAGS+=(--beta2_schedulefree <0.98>)
FLAGS+=(--batch_size <10>)
FLAGS+=(--valid_batch_size <10>)
FLAGS+=(--lr <0.01>)
FLAGS+=(--swa_lr <0.001>)
FLAGS+=(--weight_decay <5e-7>)
FLAGS+=(--lr_params_factors '{"embedding_lr_factor": 1.0, "interactions_lr_factor": 1.0, "products_lr_factor": 1.0, "readouts_lr_factor": 1.0}')
# FLAGS+=(--freeze <0>)
FLAGS+=(--amsgrad)
FLAGS+=(--scheduler "<ReduceLROnPlateau>")
FLAGS+=(--lr_factor <0.8>)
FLAGS+=(--scheduler_patience <50>)
FLAGS+=(--lr_scheduler_gamma <0.9993>)
# FLAGS+=(--swa)
# FLAGS+=(--start_swa <100>)
# FLAGS+=(--lbfgs)
# FLAGS+=(--ema)
# FLAGS+=(--ema_decay <0.99>)
FLAGS+=(--max_num_epochs <2048>)
FLAGS+=(--patience <2048>)
FLAGS+=(--eval_interval <1>)
# FLAGS+=(--keep_checkpoints)
# FLAGS+=(--save_all_checkpoints)
# FLAGS+=(--restart_latest)
# FLAGS+=(--save_cpu)
FLAGS+=(--clip_grad <10.0>)

# -------------------------------------------------------------------
# Miscellaneous
# -------------------------------------------------------------------
# FLAGS+=(--dry_run)
# FLAGS+=(--enable_cueq <false>)
# FLAGS+=(--only_cueq <false>)
# FLAGS+=(--enable_oeq <false>)
# FLAGS+=(--wandb)
# FLAGS+=(--wandb_dir "<path/to/wandb>")
# FLAGS+=(--wandb_project "<project_name>")
# FLAGS+=(--wandb_entity "<entity_name>")
# FLAGS+=(--wandb_name "<run_name>")
# FLAGS+=(--wandb_log_hypers num_channels max_L correlation lr swa_lr weight_decay batch_size max_num_epochs start_swa energy_weight forces_weight)

# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------
echo "Running: uv run mace_run train ${FLAGS[*]}"
uv run mace_run train "${FLAGS[@]}"
