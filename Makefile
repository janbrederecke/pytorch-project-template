# Set the config that is used for training
CONFIG=config
FOLDS=$(shell python3 -c "from configs.$(CONFIG) import config; print(config.FOLDS)")
GPUS=$(shell python3 -c "from configs.$(CONFIG) import config; print(config.AVAILABLE_GPUS)")
DISTRIBUTED=$(shell python3 -c "from configs.$(CONFIG) import config; print(config.DISTRIBUTED)")

# Command to use based on DISTRIBUTED
ifeq ($(DISTRIBUTED), True)
	RUN_CMD=torchrun --nproc_per_node=$(GPUS)
else
	RUN_CMD=python3
endif

# Run consecutive training on all folds
all_folds:
	@for fold in $$(seq 0 $$(($(FOLDS)-1))); do \
		$(RUN_CMD) main.py --fold $$fold --config $(CONFIG); \
	done

# Run training on first fold only
one_fold:
	$(RUN_CMD) main.py --fold 0 --config $(CONFIG);

# Run training on remaining folds after running it on fold 0
remaining_folds:
	@for fold in $$(seq 1 $$(($(FOLDS)-1))); do \
		$(RUN_CMD) main.py --fold $$fold --config $(CONFIG); \
	done

# Run training on all folds combined and validate on fold 0
all_data:
	$(RUN_CMD) main.py --fold -1 --config $(CONFIG);