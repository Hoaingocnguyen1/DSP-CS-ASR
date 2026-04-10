.PHONY: env-tts env-asr all clean

# ==========================================
# Environment 1: TTS Generation
# ==========================================
env-tts:
	@echo "Creating TTS environment (numpy 1.22.0)..."
	conda env create -f env_tts.yml

# ==========================================
# Environment 2: ASR Training & Data Prep
# ==========================================
env-asr:
	@echo "Creating ASR environment (numpy 1.26.4)..."
	conda env create -f env_asr.yml

# Create both environments
all: env-tts env-asr

# Remove environments
clean:
	conda env remove -n cs_tts -y
	conda env remove -n cs_asr -y
