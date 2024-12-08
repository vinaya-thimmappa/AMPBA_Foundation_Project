ear
:
#!/bin/bash

# Database credentials
DB_NAME="food_data"
DB_USER="ISB_FP"
DB_PASS="PennePasta1224"

# Create a new MySQL user and grant privileges
mysql -u root -e "DROP USER IF EXISTS '$DB_USER'@'localhost';"
mysql -u root -e "CREATE USER '$DB_USER'@'localhost' IDENTIFIED BY '$DB_PASS';"
mysql -u root -e "GRANT ALL PRIVILEGES ON *.* TO '$DB_USER'@'localhost' WITH GRANT OPTION;"
mysql -u root -e "FLUSH PRIVILEGES;"

# Drop the database if it exists
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

# Create the database
mysql -u $DB_USER -p$DB_PASS -e "CREATE DATABASE $DB_NAME;"
RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "Database \"$DB_NAME\" created successfully"
else
    echo "Failed to create database \"$DB_NAME\""
    exit 1
fi
