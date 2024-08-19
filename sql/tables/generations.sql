CREATE TABLE generations(
    generation_id SERIAL PRIMARY KEY,
    generator_id INT REFERENCES generators(generator_id),
    prompt_id INT REFERENCES prompt_notes(prompt_id) NULL,
    model_id INT REFERENCES models(model_id),
    generated_notes JSON,  -- generated notes
    prompt_notes JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(generator_id, prompt_id, model_id)  -- One generations per parameters
);
