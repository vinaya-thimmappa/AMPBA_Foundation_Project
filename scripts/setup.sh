#!/bin/bash

# Database credentials
DB_NAME="food_data"
DB_USER="root"
DB_PASS="PennePasta1224"

# Drop Database if exists
ERRORMSG=$(mysql -u $DB_USER -p$DB_PASS -e "DROP DATABASE IF EXISTS $DB_NAME;" 2>&1)
RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "Database \"$DB_NAME\" dropped"
else
    if [[ "$ERRORMSG" == *"Unknown database"* ]]; then
        echo "Skipping Step: Database \"$DB_NAME\" does not exist"
    else
        echo "$ERRORMSG"
        exit 1
    fi
fi

# Create Database
ERRORMSG=$(mysql -u $DB_USER -p$DB_PASS -e "CREATE DATABASE $DB_NAME;" 2>&1)
RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "Created database \"$DB_NAME\""
else
    if [[ "$ERRORMSG" == *"database exists"* ]]; then
        echo "Database \"$DB_NAME\" already exists"
    else
        echo "$ERRORMSG"
        exit 1
    fi
fi

# Grant Privileges to root
ERRORMSG=$(mysql -u $DB_USER -p$DB_PASS -e "GRANT ALL PRIVILEGES ON $DB_NAME.* TO '$DB_USER'@'localhost'; FLUSH PRIVILEGES;" 2>&1)
RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "Granted all privileges on \"$DB_NAME\" to user \"$DB_USER\""
else
    echo "$ERRORMSG"
    exit 1
fi
