PYTHON_VERSION=python3
PIP:="venv/bin/pip3"
VASTAI:="venv/bin/vastai"

.PHONY: clean pyclean

pyclean:
	@find . -name *.pyc -delete

clean: pyclean
	@rm -rf venv

build: vastai_install

venv:
	@$(PYTHON_VERSION) -m venv venv
	@$(PIP) install --upgrade pip

vastai_install: venv
	@$(PIP) install --upgrade vastai

vastai_login:
	@$(VASTAI) set api-key $(API_KEY)

vastai_offers:
	@$(VASTAI) search offers 'gpu_ram>=30 cuda_vers>=12.0' --order 'dph+' --limit 100 2>/dev/null || echo ""

vastai_instance_create:
	@$(VASTAI) search offers 'gpu_ram>=30 cuda_vers>=12.0' --order 'dph+' --limit 100 --raw | ID=$(ID) python -c "import os,sys,json; d=json.loads(sys.stdin.read()); ID=os.environ['ID']; sel=[i for i in d if int(i['id']) == int(ID)]; ([sys.stderr.write('Was not able to select offer with ID=%s\n' % ID), sys.exit(-1), ] if not sel else []); cuda_ver=str(sel[0]['cuda_max_good']); MAPPING={'12.4':'nvidia/cuda:12.3.2-devel-ubuntu20.04','12.3':'nvidia/cuda:12.3.2-devel-ubuntu20.04','12.2':'nvidia/cuda:12.2.2-devel-ubuntu20.04','12.1':'nvidia/cuda:12.1.1-devel-ubuntu20.04','12.0':'nvidia/cuda:12.0.1-devel-ubuntu20.04','11.8':'nvidia/cuda:11.8.0-devel-ubuntu20.04','11.7':'nvidia/cuda:11.7.1-devel-ubuntu20.04','11.6':'nvidia/cuda:11.6.2-devel-ubuntu20.04'}; sys.stdout.write(MAPPING[cuda_ver]);" > image_name.txt
	@echo "selecting docker image: `cat image_name.txt`" && $(VASTAI) create instance $(ID) --image `cat image_name.txt` --env 'null' --onstart-cmd 'env >> /etc/environment; touch ~/.no_auto_tmux; apt install -y python3-pip nano vim git; pip3 install numpy cupy-cuda12x matplotlib scipy dask distributed moviepy pycuda;' --disk 16 --ssh | python3 -c "import sys; inp=sys.stdin.read(); ([sys.stdout.write(inp), sys.exit(-1), ] if not inp.startswith('Started.') else []); inp=eval(inp.replace('Started.','').strip()); open('current_contract_id.txt', 'w').write(str(inp['new_contract'])); print(inp);" && $(VASTAI) ssh-url `cat current_contract_id.txt` > ssh_url.txt && rm -rf image_name.txt

vastai_instance_destroy:
	@$(VASTAI) destroy instance `cat current_contract_id.txt` && rm -rfv current_contract_id.txt && rm -rfv ssh_url.txt && rm -rfv ssh_*.json

vastai_getcode:
	@ssh -o StrictHostKeyChecking=accept-new `cat ssh_url.txt` "rm -rf electronwaves; git clone https://github.com/vincecate/electronwaves; cd electronwaves && mkdir -p results && chmod +x go.save && chmod +x makemovie.wire.nc.py;"

vastai_run:
	@ssh -o StrictHostKeyChecking=accept-new `cat ssh_url.txt` "cd electronwaves && python3 electronwaves.py $(SIMULATION_NUMBER) && bash go.save $(SIMULATION_NUMBER);"

vastai_ssh:
	@ssh -o StrictHostKeyChecking=accept-new `cat ssh_url.txt`

vastai_download:
	@scp -o StrictHostKeyChecking=accept-new `python -c 'import sys; sys.stdout.write(open("ssh_url.txt").read().replace("ssh://", "scp://"));'`/electronwaves/results/*.mp4 .
