CREATE TABLE generators (
    generator_id SERIAL PRIMARY KEY,
    generator_name VARCHAR(255),
    generator_parameters JSON,
    task VARCHAR(255)
);

CREATE UNIQUE INDEX unique_generator
ON generators (generator_name, task, (generator_parameters::text));
