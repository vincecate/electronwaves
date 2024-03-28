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
	@$(VASTAI) search offers --order 'dph+'

vastai_instance_create:
	@$(VASTAI) create instance $(ID) --image $(IMAGE) --env 'null' --onstart-cmd 'env >> /etc/environment; touch ~/.no_auto_tmux;' --disk 16 --ssh | python3 -c "import sys; inp=sys.stdin.read(); ([sys.stdout.write(inp), sys.exit(-1), ] if not inp.startswith('Started.') else []); inp=eval(inp.replace('Started.','').strip()); open('current_contract_id.txt', 'w').write(str(inp['new_contract'])); print(inp);"
	@$(VASTAI) ssh-url `cat current_contract_id.txt` > ssh_url.txt

vastai_instance_destroy:
	@$(VASTAI) destroy instance `cat current_contract_id.txt` && rm -rf current_contract_id.txt && rm -rf ssh_url.txt

vastai_run:
	@ssh `cat ssh_url.txt` "apt install python3-pip nano vim git; pip3 install numpy cupy-cuda12x matplotlib scipy dask distributed pycuda; rm -rf electronwaves; git clone https://github.com/vincecate/electronwaves; cd electronwaves && python3 electronwaves.py $(SIMULATION_NUMBER);"
