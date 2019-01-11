export PYTHONPATH=.
mode=$1
python app/app.py --mode=$mode --port=8000 --datasource=FakeTahmo

