import csv
import mysql.connector
from mysql.connector import Error



try:
    connection = mysql.connector.connect(host='localhost',
                                         database='stockmarket',
                                         user='root',
                                         password='laraiders')
    if connection.is_connected():
        db_Info = connection.get_server_info()
        print("Connected to MySQL Server version ", db_Info)
        cursor = connection.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)

except Error as e:
    print("Error while connecting to MySQL", e)



with open('aimstocks.csv', newline='') as csvfile:
     reader = csv.reader(csvfile, delimiter=',')
     for row in reader:
         print("Code: %s    Name: %s" % (row[1].strip() , row[0].strip()))

         sql = "INSERT INTO stock (symbol, name) VALUES (%s, %s)"

         cursor = connection.cursor()
         record = (row[1].strip(), row[0].strip())
         cursor.execute(sql, record)
         connection.commit()
         cursor.close()


if connection.is_connected():
    cursor.close()
    connection.close()
    print("MySQL connection is closed")