# stockmarket
Stockmarket



Install MySQL (version 8)
-------------------------
sudo apt-get install mysql-server


mysqld will log errors to /var/log/mysql/error.log
/etc/mysql/mysql.cnf

mysql -uroot -plaraiders -hlocalhost -Dstockmarket



ALTER USER 'root'@'localhost' IDENTIFIED WITH caching_sha2_password BY 'laraiders';


mysql -udebian-sys-maint -pbelhaknAIFH8OpEG

host     = localhost
user     = debian-sys-maint
password = belhaknAIFH8OpEG
socket   = /var/run/mysqld/mysqld.sock
[mysql_upgrade]
host     = localhost
user     = debian-sys-maint
password = belhaknAIFH8OpEG
socket   = /var/run/mysqld/mysqld.sock


Install Python 3
-----------------



Install Flask
-------------

pip3 install Flask
pip3 install python-dotenv


python3 -m flask --version


Python 3.8.5
Flask 2.0.1
Werkzeug 2.0.1




export FLASK_APP=app.py
python3 -m flask run

http://127.0.0.1:5000



Commands
--------

cd ./app

python3 -m flask test this-is-my-name

python3 -m flask stocks get_stock_data 1
python3 -m flask stocks get_stock_data 400
(param is starting stockid)

python3 -m flask analyse get_triple_value_stocks
python3 -m flask analyse get_triple_value_stocks_with_count
python3 -m flask analyse get_oscillating_stocks
