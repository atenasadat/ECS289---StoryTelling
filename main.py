import gpt2
import os
import requests

model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
	print(f"Downloading {model_name} model...")

	# model is saved into current directory under /models/124M/
	gpt2.download_gpt2(model_name=model_name)   


file_name = "harry_potter_novels.txt"


sess = gpt2.start_tf_sess()

# fine-tune for max steps iters
# checkpoints are by default in /checkpoint/run1
gpt2.finetune(sess,
              dataset=file_name,
              model_name='124M',
              steps=1000,
              restore_from='latest',
              run_name='run1',
              checkpoint_dir='checkpoint',
              print_every=10,
              sample_every=200,
              save_every=300
			  )

# change length to limit number of generated tokens
# change temperature (0 to 1) to control the craziness of responses
# change prefix to control the prompt
gpt2.generate(sess,
              length=50,
              temperature=0.7,
              prefix="Harry says:",
              nsamples=5,
              batch_size=5
              )