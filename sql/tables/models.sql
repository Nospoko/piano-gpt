CREATE TABLE models (
    model_id SERIAL PRIMARY KEY,
    base_model_id INT REFERENCES models(model_id) NULL,
    name VARCHAR(255),
    milion_parameters INT,
    best_val_loss FLOAT,
    train_loss FLOAT,
    iter_num INT,
    total_tokens BIGINT NULL,
    configs JSON,  -- model, dataset, data, lr, optimizer, system.dtype
    training_task VARCHAR(255), -- next_token_prediction, bass_prediction,
    wandb_link TEXT,
    created_at DATE NULL,
    UNIQUE(name, iter_num, training_task)
);
