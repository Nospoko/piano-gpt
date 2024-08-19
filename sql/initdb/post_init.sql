ALTER TABLE models
ADD COLUMN IF NOT EXISTS created_at TIMESTAMP;

-- Create or replace the function to extract date and time
CREATE OR REPLACE FUNCTION extract_datetime_from_name()
RETURNS TRIGGER AS $$
BEGIN
    NEW.created_at := TO_TIMESTAMP(
        SUBSTRING(NEW.name FROM '\d{4}-\d{2}-\d{2}-\d{2}-\d{2}'),
        'YYYY-MM-DD-HH24-MI'
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create or replace the trigger
DROP TRIGGER IF EXISTS set_created_at ON models;
CREATE TRIGGER set_created_at
BEFORE INSERT OR UPDATE ON models
FOR EACH ROW EXECUTE FUNCTION extract_datetime_from_name();

-- Update existing rows
UPDATE models
SET created_at = TO_TIMESTAMP(
    SUBSTRING(name FROM '\d{4}-\d{2}-\d{2}-\d{2}-\d{2}'),
    'YYYY-MM-DD-HH24-MI'
);
