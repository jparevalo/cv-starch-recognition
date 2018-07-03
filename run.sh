ENV=cv
source $WORKON_HOME/$ENV/bin/activate

python native.py
python processed.py
