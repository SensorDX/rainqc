export PYTHONPATH=.
mode=$1
python app/app.py --mode=$mode --datasource=FakeTahmo --startdate 01-01-2016 --enddate 12-01-2016 --trainall True

