CREATE TABLE validation_examples (
    example_id SERIAL PRIMARY KEY,
    parameters_id INT REFERENCES generation_parameters(parameters_id),
    prompt_id INT REFERENCES prompt_notes(prompt_id)
);
