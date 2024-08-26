#!/bin/bash

db_name="midi_transformers"
sql_files_path="/sql_tables"
sql_table_names=("generation_parameters" "models" "prompt_notes" "generated_notes" "validation_examples")

echo "Checking if database $db_name exists"
if psql -U "$POSTGRES_USER" -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = '$db_name';" | grep -q 1; then
    echo "Database $db_name already exists"
else
    echo "Creating database $db_name"
    psql -U "$POSTGRES_USER" -d postgres -c "CREATE DATABASE $db_name;"
fi

# Apply each SQL file to the database
for table in "${sql_table_names[@]}"; do
    echo "Applying $sql_files_path/$table.sql to $db_name"
    psql -U "$POSTGRES_USER" -d "$db_name" -f "$sql_files_path/$table.sql"
done
